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
    
    /* V7.0 신규 추가: 1:1 매칭 표(Table) 스타일 */
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
# 2-2. Gemini 응답 스키마 (V7.0 표 & 시각 분리형)
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
# 3. 보조 함수 (식약처 DB 및 구글 비전)
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
# 4. 병렬 처리 (Multi-threading) 메인 로직
# ==========================================
def process_single_chunk(idx, img_obj, ocr_extracted_text):
    try:
        model = genai.GenerativeModel(MODEL_NAME, tools=[{"google_search": {}}])
    except:
        model = genai.GenerativeModel(MODEL_NAME) 
        
    generation_config = genai.types.GenerationConfig(temperature=0.0)

    # --- [Pass 1] 원초 추출 (시각 요소 묘사 강화) ---
    extract_prompt = """객관적인 이미지 분석기입니다. 
    1. [텍스트 추출]: 이 이미지에 적힌 모든 글자를 픽셀 단위로 추출하십시오. 띄어쓰기와 출처 기호(*)를 절대 누락하지 마십시오.
    2. [시각 요소]: 요리, 원물, 연출 사진, 그릇/용기에 담긴 모습 등을 정밀하게 묘사하십시오."""
    try:
        resp1 = model.generate_content([extract_prompt, img_obj], generation_config=generation_config)
        design_raw_text = resp1.text
    except:
        design_raw_text = "텍스트 추출 실패"
    time.sleep(1)

    # --- [Pass 1.5] 정제 ---
    clean_prompt = f"다음 텍스트의 기계적 OCR 노이즈만 정제하고 띄어쓰기나 원문은 절대 바꾸지 마라:\n{design_raw_text}"
    try:
        resp15 = model.generate_content([clean_prompt], generation_config=generation_config)
        verified_text = resp15.text
    except:
        verified_text = design_raw_text

    # --- [Pass 2] V7.0 문구 1:1 매칭 & 시각 핀셋 검수 ---
    review_prompt = f"""
    당신은 대한민국 최고의 식품 마케팅 QC 선임자입니다.
    아래 [1단계 정제 데이터]와 [팩시안 데이터]를 대조하여 2가지 영역으로 나누어 정밀 분석하십시오.

    [1단계 정제 데이터 (시안)]
    {verified_text}

    [팩시안 원시 데이터 (절대 기준)]
    {ocr_extracted_text}

    📝 **[영역 1: phrase_analysis (문구별 1:1 정밀 분석)]**
    시안에 등장하는 **모든 문장, 마케팅 포인트, 성분 강조, 효능 암시 문구를 하나하나 잘게 쪼개어** 완벽히 분석하십시오.
    
    🚨 [최상위 범용 법리 통제 룰 (절대 엄수)] 🚨
    1. 🛑 **[가공상태 임의 치환 전면 금지]**: '추출액/추출물', '원액/액', '농축액/농축분말' 등은 법정 명칭입니다. 시안 마케팅 구간에서 유의어로 팩시안 명칭을 치환하면 "원재료 표기 세부 내용 불일치(🚨수정 권고)"로 적발.
    2. 🛑 **[지역명 마케팅 국가명 병기 의무]**: 우바산, 시칠리아산 등 특정 지역명 강조 시 팩시안의 '공식 국가명(스리랑카산 등)' 병기 필수.
    3. 🛑 **[부당광고 및 오인 통제]**: 
       - '낮은 카페인', '부담 없이' 등 결핍 강조 시 증빙 없으면 "수정 권고(기만)".
       - '구강 건강', '면역력' 등 신체/질병 치료 효능을 암시했으나 팩시안에 정식 인정 내용이 없으면 "치명적 위반(건기식 오인 부당광고)".
       - '소비자 조사', '병원 공동개발' 등은 팩시안이나 주석(*) 출처 필수.
    4. 🛑 **[초정밀 띄어쓰기 및 화학명]**: 하단 정보고시란의 원재료명(특히 '탄산수소나트륨' 등) 띄어쓰기 픽셀 단위 대조.
    5. **[수치 팩트체크]**: '소용량 고단백', '1,500mg' 등 수치 등장 시 팩시안 숫자를 바탕으로 수학적 계산 대조.
    6. **[면책 조항]**: "* 제품과 무관한 원물 정보" 주석이 있으면 완제품 대조는 면제(✅적합 처리).

    🖼️ **[영역 2: visual_analysis (시각 요소 및 연출컷 검증)]**
    시안에 포함된 **이미지(연출 컷, 과장된 원물, 그릇/유리잔에 담긴 형태 등)만** 집중적으로 검토하십시오.
    🚨 [시각 요소 필수 룰]
    1. 그릇, 컵, 식기 등에 제품이 덜어져 있거나 요리/원물과 함께 연출된 사진이 있다면, **무조건 '이미지 예' 또는 '조리예'라는 주의 문구가 이미지 근처에 표기되어 있는지** 확인. 누락 시 "수정 권고(소비자 오인 우려)".
    2. 특정 원물 이미지가 너무 과장되어 기만 소지가 있는지 확인.
    
    해당 구간이 완벽히 정상이라도, 무조건 "risk_level": "적합" 으로 JSON 객체를 최소 1개씩 생성하십시오.
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
                    fallback = {
                        "phrase_analysis": [{"phrase": "분석 오류", "risk_level": "수정 권고", "fact_ground": "시스템", "analysis": str(e)}],
                        "visual_analysis": [{"visual_element": "분석 오류", "risk_level": "수정 권고", "fact_ground": "시스템", "analysis": str(e)}]
                    }
                    return idx, fallback, verified_text
                time.sleep(2)

def run_parallel_analysis(main_images, ocr_extracted_text, progress_bar, status_text):
    total_chunks = len(main_images)
    final_report = {}
    chunk_list = main_images
    log_data = {}
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_chunk, i, img, ocr_extracted_text): i 
            for i, img in enumerate(main_images)
        }
        completed_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res_idx, chunk_data, verified_text = future.result()
                final_report[str(res_idx)] = chunk_data
                log_data[res_idx] = verified_text
            except Exception as e:
                st.error(f"[구간 {idx+1}] 병렬 처리 중 치명적 오류: {e}")
            
            completed_count += 1
            progress_bar.progress(completed_count / total_chunks)
            status_text.info(f"⏳ **병렬 정밀 스캔 진행 중...** [{completed_count} / {total_chunks}] 완료")
            
    total_time = time.time() - start_time
    status_text.success(f"✅ **전체 {total_chunks}개 구간 초정밀 1:1 매칭 완료!** (소요 시간: {total_time:.1f}초)")
    
    return json.dumps(final_report), chunk_list, log_data

# ==========================================
# 5. UI 및 실행 흐름
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader(
    "0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
st.sidebar.markdown("---")
uploaded_master_fact = st.sidebar.file_uploader(
    "4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True
)
st.sidebar.markdown("---")
trigger_api = st.sidebar.button("🚀 초고속 AI 핀셋 교차 검증 시작", use_container_width=True)

st.title("🛡️ 마케팅 상세페이지 정밀 통제 시스템 (V7.0 Ultimate Report)")
st.markdown("---")

# 세션 상태 초기화 (DB 팩트체크용)
if "db_targets" not in st.session_state:
    st.session_state["db_targets"] = []

if not uploaded_main_images:
    st.warning("👈 좌측 메뉴에서 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    
    # 🔥 탭(Tab) 구조 유지 (V6.2의 강력한 기능 보존)
    tab1, tab2 = st.tabs(["🛡️ 1:1 문구 정밀 분석 & 시각 요소 검증", "📊 식약처 DB 정밀 팩트체크"])
    
    # ---------------------------------------------------------
    # 탭 1: V7.0 신규 표(Table) 형식의 문구 & 이미지 분석
    # ---------------------------------------------------------
    with tab1:
        if not trigger_api:
            for img in main_img_objs:
                st.image(img, use_container_width=True)
        else:
            with st.spinner("👁️ [팩시안 기준점 확보] 구글 비전 API가 팩시안을 해독 중입니다..."):
                vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
                st.success("✅ 구글 비전 API 팩시안 판독 완료")

            st.markdown("### ⚖️ [AI 에이전트 가동] 1:1 마케팅 문구 해체 및 법규 스캔 현황")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                json_result, chunk_list, log_data = run_parallel_analysis(
                    main_img_objs, vision_extracted_text, progress_bar, status_text
                )
                report_data = json.loads(json_result)

                for idx, chunk_img in enumerate(chunk_list):
                    st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                    # 이미지는 좌측, 분석표는 우측에 넓게 배치
                    row_col1, row_col2 = st.columns([1, 2])
                    
                    with row_col1:
                        st.image(chunk_img, use_container_width=True)
                        if DEBUG_MODE and idx in log_data:
                            with st.expander("🕵️‍♂️ [디버그] Pass 1.5 정제 텍스트"):
                                st.code(log_data[idx])
                                
                    with row_col2:
                        issue_data = report_data.get(str(idx), {})
                        
                        # --- 1. 텍스트 문구 분석 표 렌더링 ---
                        st.markdown("#### 📝 텍스트 문구 1:1 정밀 분석")
                        phrase_list = issue_data.get('phrase_analysis', [])
                        if not phrase_list:
                            st.info("검토 대상 텍스트 문구가 없습니다.")
                        else:
                            html_table = "<table class='styled-table'>"
                            html_table += "<tr><th>시안 문구 (마케팅 소구점)</th><th>판정</th><th>팩시안 (법적 근거)</th><th>분석 및 사유</th></tr>"
                            for item in phrase_list:
                                risk = item.get('risk_level', '적합')
                                if risk == "치명적 위반":
                                    risk_class = "td-risk-critical"
                                elif risk == "수정 권고":
                                    risk_class = "td-risk-warning"
                                else:
                                    risk_class = "td-risk-pass"
                                    
                                html_table += f"<tr>"
                                html_table += f"<td class='td-phrase'>{item.get('phrase', '')}</td>"
                                html_table += f"<td class='{risk_class}'>{risk}</td>"
                                html_table += f"<td class='td-fact'>{item.get('fact_ground', '')}</td>"
                                html_table += f"<td class='td-analysis'>{item.get('analysis', '')}</td>"
                                html_table += "</tr>"
                            html_table += "</table>"
                            st.markdown(html_table, unsafe_allow_html=True)

                        # --- 2. 시각 요소 전용 분석 렌더링 ---
                        st.markdown("#### 🖼️ 시각 요소 및 연출 컷 분석")
                        visual_list = issue_data.get('visual_analysis', [])
                        if not visual_list:
                            st.info("시각적 특이사항이 감지되지 않았습니다.")
                        else:
                            for item in visual_list:
                                risk = item.get('risk_level', '적합')
                                if risk == "치명적 위반":
                                    css_class, icon = "risk-critical", "❌ [치명적 위반]"
                                elif risk == "수정 권고":
                                    css_class, icon = "risk-warning", "⚠️ [수정 권고]"
                                else:
                                    css_class, icon = "risk-pass", "✅ [적합]"

                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f"**{icon} 타겟 요소:** {item.get('visual_element', '')}")
                                st.markdown(f"- **판정 사유:** {item.get('analysis', '')}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                    st.markdown("---")
            except Exception as e:
                st.error(f"전체 프로세스 중 오류 발생: {e}")

    # ---------------------------------------------------------
    # 탭 2: 식약처 DB 정밀 매칭 (V6.2 핵심 로직 100% 보존)
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### 📊 식약처 DB 출처 팩트체크 전용 스캐너")
        st.info("상세페이지 내에 '식약처 식품영양성분 DB' 등을 출처로 타 원물(소고기, 닭고기 등)의 수치를 비교한 부분을 정밀하게 추출하고, 실제 식약처 API와 1:1 매칭하여 검증합니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("1단계: 출처가 명시된 원물 수치 추출", use_container_width=True):
                with st.spinner("이미지 전체를 스캔하여 DB 비교 타겟을 찾는 중입니다..."):
                    model = genai.GenerativeModel(MODEL_NAME)
                    payload = [img.resize((1000, int(img.size[1] * (1000.0 / img.size[0]))), Image.LANCZOS) if img.size[0] > 1000 else img for img in main_img_objs]
                    
                    extract_prompt = """
                    이 16장의 상세페이지 이미지들을 모두 확인하여, 하단이나 주변에 '식약처 DB', '국가표준식품성분표' 등을 출처로 명시하고 있는 인포그래픽(비교 수치)이 있는지 샅샅이 찾아내십시오.
                    발견되었다면 해당 수치들을 아래 JSON 배열 형태로만 출력하십시오. 없으면 빈 배열 [] 을 출력하십시오.
                    
                    [JSON 출력 양식]
                    [
                        {
                            "기본명사": "쇠고기", 
                            "상세상태": "한우 등심 구운것", 
                            "표기된수치": "18.9g"
                        },
                        {
                            "기본명사": "대두", 
                            "상세상태": "노란콩 말린것", 
                            "표기된수치": "36.2g"
                        }
                    ]
                    """
                    payload.append(extract_prompt)
                    
                    try:
                        res = model.generate_content(payload, generation_config=genai.types.GenerationConfig(temperature=0.0)).text.strip()
                        targets = json.loads(re.sub(r'```json\s*|```\s*', '', res))
                        st.session_state["db_targets"] = targets
                        st.success(f"🎯 총 {len(targets)}개의 검증 타겟을 추출했습니다!")
                    except Exception as e:
                        st.error(f"타겟 추출 실패: {e}")

        with col2:
            if st.button("2단계: 식약처 DB 정밀 매칭 및 검증", use_container_width=True):
                if not st.session_state.get("db_targets"):
                    st.warning("먼저 '1단계' 버튼을 눌러 검증할 타겟을 추출해 주십시오.")
                else:
                    with st.spinner("식약처 API와 통신하여 세부 조리법 데이터를 대조 중입니다..."):
                        for target in st.session_state["db_targets"]:
                            base_noun = target.get("기본명사", "")
                            detail_state = target.get("상세상태", "")
                            claimed_val = target.get("표기된수치", "")
                            
                            st.markdown(f"#### 🔎 타겟: `{detail_state}` (시안 표기값: {claimed_val})")
                            
                            db_data = query_food_nutrient_db(base_noun)
                            if not db_data:
                                st.error(f"식약처 DB에서 '{base_noun}'에 대한 검색 결과를 찾지 못했습니다.")
                                continue
                                
                            simplified_db = [
                                f"- [{row.get('DESC_KOR', '이름없음')}] 단백질:{row.get('NUTR_CONT3')}g, 지방:{row.get('NUTR_CONT4')}g, 열량:{row.get('NUTR_CONT1')}kcal"
                                for row in db_data[:50]
                            ]
                            db_list_text = "\n".join(simplified_db)
                            
                            model = genai.GenerativeModel(MODEL_NAME)
                            match_prompt = f"""
                            당신은 식품 DB 팩트체커입니다.
                            시안에 표기된 목표 원물은 **[{detail_state}]** 이고, 단백질 등의 강조 수치는 **[{claimed_val}]** 입니다.
                            
                            아래는 식약처 API에서 '{base_noun}'로 검색한 50개의 목록입니다.
                            이 목록을 샅샅이 뒤져서 **[{detail_state}]**와 가장 일치하는 부위/조리법 항목을 하나 찾아내십시오.
                            그리고 그 항목의 수치와 시안의 수치({claimed_val})가 일치하는지 판정하여 마크다운 표로 깔끔하게 출력하십시오.
                            
                            [식약처 DB 목록]
                            {db_list_text}
                            """
                            
                            try:
                                match_res = model.generate_content([match_prompt], generation_config=genai.types.GenerationConfig(temperature=0.0)).text
                                st.markdown(match_res)
                            except Exception as e:
                                st.error(f"매칭 연산 실패: {e}")
                            
                            st.markdown("---")
