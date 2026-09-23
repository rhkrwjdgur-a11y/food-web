import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import os
import json
import time
import requests
import urllib.parse
import re
import socket

# ==========================================
# 1. 기본 페이지 설정 및 네트워크 방어
# ==========================================
st.set_page_config(page_title="식품 상세페이지 QC 마스터", layout="wide")
socket.setdefaulttimeout(600)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545; margin-bottom: 10px; }
    .risk-warning { background-color: #fefaf0; padding: 15px; border-radius: 5px; border-left: 5px solid #f39c12; margin-bottom: 10px; }
    .risk-pass { background-color: #f4fbf7; padding: 15px; border-radius: 5px; border-left: 5px solid #2ecc71; margin-bottom: 10px; }
    .styled-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.95em; box-shadow: 0 0 10px rgba(0, 0, 0, 0.05); background-color: white;}
    .styled-table th { background-color: #343a40; color: white; text-align: center; padding: 12px; }
    .styled-table td { border: 1px solid #dee2e6; padding: 10px; vertical-align: top; }
    .td-phrase { font-weight: bold; color: #0056b3; width: 25%; }
    .td-risk-critical { background-color: #ffe3e3; color: #c92a2a; font-weight: bold; text-align: center; width: 12%; }
    .td-risk-warning { background-color: #fff3cd; color: #b08d00; font-weight: bold; text-align: center; width: 12%; }
    .td-risk-pass { background-color: #d3f9d8; color: #2b8a3e; font-weight: bold; text-align: center; width: 12%; }
    .td-fact { font-size: 0.9em; color: #495057; width: 25%; background-color: #f8f9fa; }
    .td-analysis { width: 38%; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. API 키 연동 (Secrets)
# ==========================================
try:
    genai.configure(api_key=st.secrets["AI_VISION_API_KEY"])
    FOOD_API_KEY = st.secrets["FOOD_SAFETY_API_KEY"]

    gcp_json_string = st.secrets["gcp_service_account"]["GOOGLE_VISION_KEY"]
    gcp_credentials = json.loads(gcp_json_string)
    gcp_credentials["private_key"] = gcp_credentials["private_key"].replace("\\n", "\n")

    vision_credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    vision_client = vision.ImageAnnotatorClient(credentials=vision_credentials)

except KeyError as e:
    st.error(f"시스템 오류: Secrets 설정 누락 - {e}")
    st.stop()
except Exception as e:
    st.error(f"구글 인증 오류: {e}")
    st.stop()

DEBUG_MODE = st.sidebar.checkbox("🐞 디버그 모드 및 시스템 로그", value=True)
MODEL_NAME = "gemini-2.5-flash"

# ==========================================
# 2-2. Gemini 응답 스키마
# ==========================================
REVIEW_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "phrase_analysis": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "phrase": {"type": "STRING"},
                    "risk_level": {"type": "STRING", "enum": ["치명적 위반", "수정 권고", "적합"]},
                    "fact_ground": {"type": "STRING"},
                    "analysis": {"type": "STRING"}
                },
                "required": ["phrase", "risk_level", "fact_ground", "analysis"]
            }
        },
        "visual_analysis": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "visual_element": {"type": "STRING"},
                    "risk_level": {"type": "STRING", "enum": ["치명적 위반", "수정 권고", "적합"]},
                    "fact_ground": {"type": "STRING"},
                    "analysis": {"type": "STRING"}
                },
                "required": ["visual_element", "risk_level", "fact_ground", "analysis"]
            }
        }
    },
    "required": ["phrase_analysis", "visual_analysis"]
}

# ==========================================
# 3. 보조 함수
# ==========================================
@st.cache_data(show_spinner=False)
def extract_text_with_google_vision(uploaded_files):
    if not uploaded_files: return "팩시안 이미지 없음"
    extracted_text = ""
    for file in uploaded_files:
        content = file.read()
        image = vision.Image(content=content)
        response = vision_client.document_text_detection(image=image)
        if response.full_text_annotation:
            extracted_text += response.full_text_annotation.text + "\n\n"
        file.seek(0)
    return extracted_text

def query_food_nutrient_db(food_name):
    if not food_name or not FOOD_API_KEY: return None
    std_dict = {"쇠고기": "소고기", "계육": "닭고기", "돈육": "돼지고기"}
    search_name = std_dict.get(food_name.strip(), food_name.strip())
    url = f"http://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02?serviceKey={FOOD_API_KEY}&pageNo=1&numOfRows=200&type=json&DESC_KOR={urllib.parse.quote(search_name)}"
    try:
        res_json = json.loads(requests.get(url, timeout=15).text.strip())
        body = res_json.get('body') or res_json.get('response', {}).get('body', {})
        items = body.get('items', [])
        return items['item'] if isinstance(items, dict) and 'item' in items else items if isinstance(items, list) else [items]
    except: return None

# ==========================================
# 4. 병렬 처리 로직
# ==========================================
def process_single_chunk(idx, img_obj, ocr_extracted_text):
    try:
        model = genai.GenerativeModel(MODEL_NAME, tools=[{"google_search": {}}])
    except:
        model = genai.GenerativeModel(MODEL_NAME) 
        
    generation_config = genai.types.GenerationConfig(temperature=0.0)

    extract_prompt = """객관적인 이미지 분석기입니다. 
    1. [텍스트 추출]: 이 이미지에 적힌 모든 글자를 픽셀 단위로 추출. 띄어쓰기, 기호 절대 누락 금지.
    2. [시각 요소]: 요리, 원물, 연출 사진, 그릇/용기에 담긴 모습 정밀 묘사."""
    try:
        resp1 = model.generate_content([extract_prompt, img_obj], generation_config=generation_config)
        design_raw_text = resp1.text
    except:
        design_raw_text = "텍스트 추출 실패"
    time.sleep(1)

    clean_prompt = f"다음 텍스트의 기계적 OCR 노이즈만 정제하고 띄어쓰기나 원문은 절대 바꾸지 마라:\n{design_raw_text}"
    try:
        resp15 = model.generate_content([clean_prompt], generation_config=generation_config)
        verified_text = resp15.text
    except:
        verified_text = design_raw_text

    review_prompt = f"""
    당신은 대한민국 최고의 식품 마케팅 QC 선임자입니다.
    아래 [1단계 정제 데이터(시안)]와 [팩시안 데이터]를 대조하여 2가지 영역으로 정밀 분석하십시오.

    [1단계 정제 데이터 (시안)]
    {verified_text}

    [팩시안 원시 데이터 (절대 기준)]
    {ocr_extracted_text}

    📝 **[영역 1: phrase_analysis (문구별 1:1 정밀 분석)]**
    시안에 등장하는 모든 문장, 마케팅 포인트를 잘게 쪼개어 완벽히 분석하십시오.
    🚨 [최상위 법리 룰]:
    1. [가공상태 임의 치환 금지]: 추출액을 추출물로 치환하는 등 임의 변경 적발.
    2. [국가명 병기]: 우바산 등 지역명 단독 표기 불가(스리랑카산 병기).
    3. [부당광고]: 낮은 카페인, 구강 건강 등 결핍/효능 암시 시 증빙 없으면 치명적 위반 처리. 조사결과(소비자조사 등) 출처 누락 시 지적.
    4. [띄어쓰기]: 탄산수소나트륨 등 화학명 띄어쓰기 검수.

    🖼️ **[영역 2: visual_analysis (시각 요소 전용 검증)]**
    🚨 [시각 룰]:
    1. 그릇, 컵 등에 덜어져 있거나 원물 연출 시 '이미지 예', '조리예' 문구 누락 확인.
    
    해당 구간이 정상이라도 "risk_level": "적합" 객체를 생성하십시오.
    """

    for attempt in range(3):
        try:
            review_response = model.generate_content(
                [review_prompt, img_obj],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=REVIEW_RESPONSE_SCHEMA,
                ),
            )
            return idx, json.loads(review_response.text), verified_text
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                time.sleep(5)
            else:
                if attempt == 2: 
                    return idx, {"phrase_analysis": [{"phrase": "오류", "risk_level": "수정 권고", "fact_ground": "시스템", "analysis": str(e)}], "visual_analysis": []}, verified_text
                time.sleep(2)

def run_parallel_analysis(main_images, ocr_extracted_text, progress_bar, status_text):
    total_chunks = len(main_images)
    final_report = {}
    chunk_list = main_images
    log_data = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_chunk, i, img, ocr_extracted_text): i for i, img in enumerate(main_images)}
        completed_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            res_idx, chunk_data, verified_text = future.result()
            final_report[str(res_idx)] = chunk_data
            log_data[res_idx] = verified_text
            
            completed_count += 1
            progress_bar.progress(completed_count / total_chunks)
            status_text.info(f"⏳ **병렬 정밀 스캔 진행 중...** [{completed_count} / {total_chunks}] 완료")
            
    status_text.success(f"✅ **전체 {total_chunks}개 구간 초정밀 1:1 매칭 완료!** (소요 시간: {time.time() - start_time:.1f}초)")
    return json.dumps(final_report), chunk_list, log_data

# ==========================================
# 5. UI 및 실행 흐름 (세션 상태 안정성 확보)
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.sidebar.markdown("---")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 (팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
st.sidebar.markdown("---")
trigger_api = st.sidebar.button("🚀 초고속 AI 핀셋 교차 검증 시작", use_container_width=True)

st.title("🛡️ 마케팅 상세페이지 정밀 통제 시스템 (V7.1 Pro)")
st.markdown("---")

# 세션 관리 (탭 이동 시 초기화 방지)
if "db_targets" not in st.session_state: st.session_state["db_targets"] = []
if "analysis_done" not in st.session_state: st.session_state["analysis_done"] = False

if trigger_api and uploaded_main_images:
    st.session_state["analysis_done"] = True
    st.session_state.pop("full_report_data", None) # 새 검증 시작 시 기존 데이터 초기화

if not uploaded_main_images:
    st.warning("👈 좌측 메뉴에서 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    
    # 🔥 3개의 완벽한 워크플로우 탭
    tab1, tab2, tab3 = st.tabs(["🛡️ 1:1 문구/시각 정밀 분석", "📊 식약처 DB 정밀 팩트체크", "✉️ 마케팅팀 전달용 리포트"])
    
    # ---------------------------------------------------------
    # 탭 1: 표(Table) 형식의 문구 & 이미지 분석
    # ---------------------------------------------------------
    with tab1:
        if not st.session_state["analysis_done"]:
            for img in main_img_objs:
                st.image(img, use_container_width=True)
        else:
            if "full_report_data" not in st.session_state:
                with st.spinner("👁️ [팩시안 기준점 확보] 구글 비전 API가 팩시안을 해독 중입니다..."):
                    vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
                
                st.markdown("### ⚖️ [AI 에이전트 가동] 1:1 마케팅 문구 해체 및 법규 스캔 현황")
                progress_bar = st.progress(0)
                status_text = st.empty()

                json_result, chunk_list, log_data = run_parallel_analysis(
                    main_img_objs, vision_extracted_text, progress_bar, status_text
                )
                
                st.session_state["full_report_data"] = json.loads(json_result)
                st.session_state["chunk_list"] = chunk_list
                st.session_state["log_data"] = log_data

            # 세션에 저장된 데이터로 화면 렌더링 (탭 이동 시 보존됨)
            report_data = st.session_state["full_report_data"]
            chunk_list = st.session_state["chunk_list"]
            log_data = st.session_state["log_data"]

            for idx, chunk_img in enumerate(chunk_list):
                st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                row_col1, row_col2 = st.columns([1, 2])
                
                with row_col1:
                    st.image(chunk_img, use_container_width=True)
                    if DEBUG_MODE and idx in log_data:
                        with st.expander("🕵️‍♂️ [디버그] Pass 1.5 정제 텍스트"):
                            st.code(log_data[idx])
                            
                with row_col2:
                    issue_data = report_data.get(str(idx), {})
                    
                    st.markdown("#### 📝 텍스트 문구 1:1 정밀 분석")
                    phrase_list = issue_data.get('phrase_analysis', [])
                    if not phrase_list:
                        st.info("검토 대상 텍스트 문구가 없습니다.")
                    else:
                        html_table = "<table class='styled-table'>"
                        html_table += "<tr><th>시안 문구 (마케팅 소구점)</th><th>판정</th><th>팩시안 (법적 근거)</th><th>분석 및 사유</th></tr>"
                        for item in phrase_list:
                            risk = item.get('risk_level', '적합')
                            risk_class = "td-risk-critical" if risk == "치명적 위반" else "td-risk-warning" if risk == "수정 권고" else "td-risk-pass"
                            html_table += f"<tr><td class='td-phrase'>{item.get('phrase', '')}</td><td class='{risk_class}'>{risk}</td><td class='td-fact'>{item.get('fact_ground', '')}</td><td class='td-analysis'>{item.get('analysis', '')}</td></tr>"
                        html_table += "</table>"
                        st.markdown(html_table, unsafe_allow_html=True)

                    st.markdown("#### 🖼️ 시각 요소 및 연출 컷 분석")
                    visual_list = issue_data.get('visual_analysis', [])
                    if not visual_list:
                        st.info("시각적 특이사항이 감지되지 않았습니다.")
                    else:
                        for item in visual_list:
                            risk = item.get('risk_level', '적합')
                            css_class, icon = ("risk-critical", "❌") if risk == "치명적 위반" else ("risk-warning", "⚠️") if risk == "수정 권고" else ("risk-pass", "✅")
                            st.markdown(f'<div class="{css_class}">**{icon} 타겟 요소:** {item.get("visual_element", "")}<br>- **판정 사유:** {item.get("analysis", "")}</div>', unsafe_allow_html=True)
                st.markdown("---")

    # ---------------------------------------------------------
    # 탭 2: 식약처 DB 정밀 매칭
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### 📊 식약처 DB 출처 팩트체크 전용 스캐너")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("1단계: 출처 명시 원물 수치 추출", use_container_width=True):
                with st.spinner("DB 비교 타겟 탐색 중..."):
                    model = genai.GenerativeModel(MODEL_NAME)
                    payload = [img.resize((1000, int(img.size[1]*(1000.0/img.size[0]))), Image.LANCZOS) if img.size[0]>1000 else img for img in main_img_objs]
                    payload.append("""시안 전체에서 '식약처 DB' 등 출처 표기 인포그래픽 탐색. JSON 배열로 출력.
                    [{"기본명사": "쇠고기", "상세상태": "한우 등심 구운것", "표기된수치": "18.9g"}]""")
                    try:
                        res = model.generate_content(payload, generation_config=genai.types.GenerationConfig(temperature=0.0)).text.strip()
                        st.session_state["db_targets"] = json.loads(re.sub(r'```json\s*|```\s*', '', res))
                        st.success(f"🎯 {len(st.session_state['db_targets'])}개 추출 완료!")
                    except Exception as e:
                        st.error(f"추출 실패: {e}")

        with col2:
            if st.button("2단계: 식약처 DB 정밀 매칭", use_container_width=True):
                if not st.session_state.get("db_targets"):
                    st.warning("1단계를 먼저 실행하세요.")
                else:
                    for target in st.session_state["db_targets"]:
                        b_noun, d_state, c_val = target.get("기본명사", ""), target.get("상세상태", ""), target.get("표기된수치", "")
                        st.markdown(f"#### 🔎 `{d_state}` (시안: {c_val})")
                        db_data = query_food_nutrient_db(b_noun)
                        if not db_data:
                            st.error("결과 없음"); continue
                        sim_db = [f"- [{r.get('DESC_KOR','명칭없음')}] 단백질:{r.get('NUTR_CONT3')}g" for r in db_data[:50]]
                        try:
                            m_res = genai.GenerativeModel(MODEL_NAME).generate_content([f"타겟:[{d_state}], 수치:[{c_val}]\n아래 50개 목록에서 찾아 대조표 작성.\n" + "\n".join(sim_db)]).text
                            st.markdown(m_res)
                        except:
                            st.error("연산 실패")
                        st.markdown("---")

    # ---------------------------------------------------------
    # 탭 3: 마케팅팀 전달용 최종 리포트 자동 생성
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### ✉️ 마케팅팀 전달용 종합 리포트 생성기")
        st.info("표시팀 담당자로서 탭 1에서 도출된 '치명적 위반' 및 '수정 권고' 사항들만 모아, 메신저 복사/붙여넣기용 최종 지시사항을 생성합니다.")
        
        if "full_report_data" not in st.session_state:
            st.warning("먼저 '탭 1'에서 교차 검증 스캔을 완료해 주십시오.")
        else:
            if st.button("📝 마케팅팀 전달용 수정사항 종합하기", use_container_width=True):
                with st.spinner("표시팀 담당자 모드로 수정사항을 요약 및 포맷팅 중입니다..."):
                    model = genai.GenerativeModel(MODEL_NAME)
                    report_prompt = f"""
                    당신은 식품 제조사 품질표시팀 담당자입니다.
                    아래 제공된 [QC 1:1 정밀 검토 결과]를 분석하여, 마케팅팀에게 이메일이나 메신저로 바로 전달할 수 있는 '최종 수정요청 종합 리포트'를 작성하십시오.

                    [작성 절대 규칙]
                    1. '적합' 판정을 받은 내용은 모조리 버리고, 오직 '수정 권고' 및 '치명적 위반' 판정을 받은 문제점만 모아서 작성하십시오.
                    2. 인사말이나 서론은 일절 생략하고, 즉시 `<수정사항>`이라는 제목으로 시작하십시오.
                    3. 마케팅팀이 보고 바로 디자인 파일 원본을 고칠 수 있도록 "어떤 단어를 -> 어떻게 바꾸어라", "무엇을 삭제하라", "어디에 어떤 문구를 추가하라"는 식으로 아래 [참고 양식]과 100% 동일한 직관적인 텍스트 형태로 지시하십시오.
                    4. 법적 위반 소지(부당광고, 원재료명 불일치 등)가 있는 경우, 예시처럼 콜론(:) 뒤에 해당 법적 근거를 간결하고 단호하게 명시하십시오.

                    [참고 양식]
                    <수정사항>
                    1. (메인페이지) 우바산 홍차 → 스리랑카산 우바 홍차
                    2. (목차페이지) 낮은카페인 (삭제) : 「식품등의 표시기준」 Ⅲ. 1. 자. 2)에 따라 다류는 90% 이상 제거해야 디카페인 표기 가능. 해당 문구는 부당광고 소지.
                    3. (제품소개) 원재료명 : 탄산수 소나트륨 → 탄산수소나트륨 (붙여쓰기)
                    4. (연출컷) 컵에 담긴 제품 이미지 주변에 '이미지 예' 혹은 '조리예' 문구 추가

                    [QC 1:1 정밀 검토 결과 (JSON)]
                    {json.dumps(st.session_state["full_report_data"], ensure_ascii=False)}
                    """
                    
                    try:
                        final_report_res = model.generate_content(
                            [report_prompt], 
                            generation_config=genai.types.GenerationConfig(temperature=0.2)
                        ).text
                        st.markdown("#### 📋 복사 후 바로 전송하세요")
                        st.code(final_report_res, language='markdown')
                    except Exception as e:
                        st.error(f"리포트 생성 중 오류 발생: {e}")
