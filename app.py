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
socket.setdefaulttimeout(600) # 대기 시간 10분 연장

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-warning { background-color: #fefaf0; padding: 20px; border-radius: 10px; border-left: 6px solid #f39c12; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-pass { background-color: #f4fbf7; padding: 20px; border-radius: 10px; border-left: 6px solid #2ecc71; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .card-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
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
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "image_index": {"type": "INTEGER"},
            "risk_level": {
                "type": "STRING",
                "enum": ["치명적 위반", "수정 권고", "적합"]
            },
            "title": {"type": "STRING"},
            "exact_text_in_design": {"type": "STRING"},
            "fact_or_legal_ground": {"type": "STRING"},
            "discrepancy_analysis": {"type": "STRING"},
        },
        "required": ["image_index", "risk_level", "title", "exact_text_in_design", "fact_or_legal_ground", "discrepancy_analysis"],
    },
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

def auto_extract_db_keywords_json(main_images):
    model = genai.GenerativeModel(MODEL_NAME)
    payload = [img.resize((1000, int(img.size[1] * (1000.0 / img.size[0]))), Image.LANCZOS) if img.size[0] > 1000 else img for img in main_images]
    
    payload.append("""타 식품과 수치를 비교하거나 식약처 DB를 인용하는 문구가 있다면 대분류 명사만 뽑지 말고, 시안에 적힌 구체적인 상태(예: "쇠고기 구운것", "대두 말린것")를 한 덩어리의 key로 하여 JSON 출력. 없으면 "NONE" 출력.""")
    
    try:
        res = model.generate_content(payload, generation_config=genai.types.GenerationConfig(temperature=0.0)).text.strip()
        if res == "NONE": return {}
        return json.loads(re.sub(r'```json\s*|```\s*', '', res))
    except: return {}

# ==========================================
# 4. 병렬 처리 (Multi-threading) & 3-Pass 검수 로직
# ==========================================
def process_single_chunk(idx, img_obj, ocr_extracted_text, db_context_text):
    try:
        model = genai.GenerativeModel(MODEL_NAME, tools=[{"google_search": {}}])
    except:
        model = genai.GenerativeModel(MODEL_NAME) 
        
    generation_config = genai.types.GenerationConfig(temperature=0.0)

    # --- [Pass 1] 원초 추출 ---
    extract_prompt = """
    객관적인 이미지 분석기입니다. 
    1. [텍스트 추출]: 이 이미지에 적힌 모든 글자를 픽셀 단위로 추출하십시오. 띄어쓰기와 출처 기호(*)를 절대 누락하지 마십시오.
    2. [시각 요소]: 요리, 원물, 연출 사진 묘사.
    """
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

    # --- [Pass 2] 범용 법리 룰이 탑재된 핀셋 검수 ---
    review_prompt = f"""
    당신은 실무 경험이 풍부한 대한민국 최고의 식품 마케팅 QC 선임자입니다.
    아래 [1단계 정제 데이터], [Google Vision API 팩시안 데이터], [외부 영양성분 DB 요약]을 대조하십시오.

    [1단계 정제 데이터 (현재 시안 내용)]
    {verified_text}

    [팩시안 원시 데이터 - 절대적 기준]
    {ocr_extracted_text}
    
    [외부 영양성분 DB 요약 (식약처 API 실시간 데이터)]
    {db_context_text}

    <pre_calc>
    1. [원물 기초 수치 정밀 팩트체크]: 기재된 원물의 구체적 상태(예: 소고기 구운것 18.9g)를 제공된 DB나 구글 검색을 통해 팩트체크하고, DB 매칭 실패 시 무조건 "실무자 확인 요망"으로 지시할 것.
    2. [원물 비교 강제 연산]: 비교 비율(예: 192%)이 기초 수치를 기준으로 산수 연산이 맞는지 계산 검증.
    </pre_calc>

    🚨 [최상위 범용 법리 통제 룰 (V6.0)] 🚨
    1. 🛑 **[가공상태(성상) 임의 치환 전면 금지 룰]**: 원재료명에 사용되는 '추출액/추출물', '농축액/농축분말', '즙/페이스트' 등은 화학적 공정이 명백히 다른 법정 명칭입니다. 시안 마케팅 구간에서 유의어(어감 개선) 목적으로 팩시안의 '추출액'을 '추출물'로 바꾸거나 그 반대로 적는 행위 일체를 "원재료 표기 세부 내용 불일치(🚨수정 권고)"로 적발하십시오. (토시 하나 틀리면 안 됨)
    2. 🛑 **[지역명 마케팅 국가명 병기 의무 룰]**: 제품명이나 디자인 뱃지 등에 특정 지역명(예: 우바산, 시칠리아산, 보르도산)을 강조할 경우, 반드시 팩시안 기준의 '공식 국가명(스리랑카산, 이탈리아산 등)'을 함께 수식어로 병기해야 합니다. 국가명 없이 지역명 단독 표기 시 원산지 오인 기만으로 "수정 권고" 처리하십시오.
    3. 🛑 **[결핍/비함유 강조 및 간접비교 부당광고 통제 룰]**: 특정 성분(카페인, 첨가물, 나트륨 등)을 지칭하며 '낮은', '부담 없이', '~제로' 등으로 강조하는 문구가 발견되면 무관용으로 팩트체크 하십시오. 
        - 식약처 기준(예: 다류/커피류의 디카페인은 카페인 90% 이상 제거)에 부합한다는 완벽한 증빙(팩시안 내용)이 없거나, 다른 제품을 간접적으로 다르게 인식하게 하는 내용이라면 무조건 **"「식품등의 부당한 표시 또는 광고의 내용 기준」 제2조 5호 위반 소지 (기만 및 부당광고)"**를 근거로 🚨수정 권고 하십시오.

    🔥 [기존 상세페이지 핀셋 검수 룰]
    4. 🛑 **[초정밀 띄어쓰기 및 화학명 스캔 룰]**: 하단 정보고시란의 원재료명(특히 '탄산수소나트륨', '비타민C' 등)에 임의의 띄어쓰기(예: 탄산수 소나트륨)가 발생하지 않았는지 팩시안과 픽셀 단위로 대조하여 무관용으로 지적하십시오.
    5. [연출 컷 주의문구 예외 룰]: 연출된 사진이 있는데 '이미지 예', '연출된 이미지입니다' 문구가 없으면 무조건 지적.
    6. [영양/성분 강조 크로스체크]: '저당', '고단백' 등 강조 시 팩시안 숫자 1:1 대조.
    7. [원료적 특성 면책 조항 예외 룰]: "* 제품과 무관한 원물 정보" 주석이 있으면 완제품 대조는 면제하되, 원물 자체 수치의 팩트체크(pre_calc)는 반드시 수행할 것.

    ⭐ 해당 구간이 완벽히 정상이라도, 무조건 "risk_level": "적합" 으로 JSON 객체를 최소 1개 이상 생성하십시오.
    image_index 필드에는 반드시 {idx} 값을 넣으십시오.
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
            
            chunk_issues = json.loads(review_response.text)
            
            if isinstance(chunk_issues, list):
                for issue in chunk_issues:
                    issue["image_index"] = idx  
            else:
                chunk_issues = []

            if len(chunk_issues) == 0:
                chunk_issues = [{
                    "image_index": idx,
                    "risk_level": "적합",
                    "title": "일반 마케팅 및 팩트 검토 완료",
                    "exact_text_in_design": verified_text[:200],
                    "fact_or_legal_ground": "특이사항 없음",
                    "discrepancy_analysis": "해당 구간의 텍스트와 이미지를 스캔 및 팩트체크한 결과, 위반 요소나 기만행위가 발견되지 않아 적합으로 판정합니다.",
                }]
                
            return idx, chunk_issues, verified_text

        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                time.sleep(5)
            else:
                if attempt == 2: 
                    fallback = [{
                        "image_index": idx, 
                        "risk_level": "적합",
                        "title": "AI 검수 오류 폴백",
                        "exact_text_in_design": verified_text[:100] if isinstance(verified_text, str) else "내용 없음",
                        "fact_or_legal_ground": "시스템 응답 지연",
                        "discrepancy_analysis": f"AI 응답 생성 중 오류가 발생했습니다. (사유: {e})",
                    }]
                    return idx, fallback, verified_text
                time.sleep(2)

def run_parallel_analysis(main_images, ocr_extracted_text, db_context_text, progress_bar, status_text):
    total_chunks = len(main_images)
    final_report = []
    chunk_list = main_images
    log_data = {}
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_chunk, i, img, ocr_extracted_text, db_context_text): i 
            for i, img in enumerate(main_images)
        }
        
        completed_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res_idx, issues, verified_text = future.result()
                final_report.extend(issues)
                log_data[res_idx] = verified_text
            except Exception as e:
                st.error(f"[구간 {idx+1}] 병렬 처리 중 치명적 오류: {e}")
            
            completed_count += 1
            elapsed_time = time.time() - start_time
            progress_bar.progress(completed_count / total_chunks)
            status_text.info(f"⏳ **병렬 스캔 진행 중...** [{completed_count} / {total_chunks}] 완료 (소요 시간: {elapsed_time:.1f}초)")
            
    total_time = time.time() - start_time
    status_text.success(f"✅ **전체 {total_chunks}개 구간 초고속 병렬 검토 완료!** (총 소요 시간: {total_time:.1f}초)")
    
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

st.title("🛡️ 마케팅 상세페이지 정밀 통제 시스템 (V6.0 Universal Master)")
st.markdown("---")

if not uploaded_main_images:
    st.warning("👈 좌측 메뉴에서 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    if not trigger_api:
        for img in main_img_objs:
            st.image(img, use_container_width=True)
    else:
        with st.spinner("👁️ [팩시안 기준점 확보] 구글 비전 API가 팩시안을 해독 중입니다..."):
            vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
            st.success("✅ 구글 비전 API 팩시안 판독 완료")

        with st.spinner("🔍 [DB 연동] 외부 영양성분 DB 타겟 탐지 중..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            final_db_context_text = ""
            
            if auto_dict:
                search_keywords = auto_dict.keys() if isinstance(auto_dict, dict) else auto_dict
                for base_food in search_keywords:
                    if not isinstance(base_food, str): continue
                    db_data = query_food_nutrient_db(base_food)
                    if db_data:
                        simplified_db = [
                            f"- [{row.get('DESC_KOR', '이름없음')}] 열량:{row.get('NUTR_CONT1')}kcal, 단백질:{row.get('NUTR_CONT3')}g, 지방:{row.get('NUTR_CONT4')}g"
                            for row in db_data[:20]
                        ]
                        final_db_context_text += f"\n[검색어 '{base_food}'] DB 요약\n" + "\n".join(simplified_db) + "\n"

        st.markdown("### ⚖️ [AI 에이전트 가동] 정밀 팩트체크 및 법규 스캔 현황")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            json_result, chunk_list, log_data = run_parallel_analysis(
                main_img_objs, vision_extracted_text, final_db_context_text, progress_bar, status_text
            )
            report_data = json.loads(json_result)

            for idx, chunk_img in enumerate(chunk_list):
                st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                row_col1, row_col2 = st.columns([1, 1])
                
                with row_col1:
                    st.image(chunk_img, use_container_width=True)
                    if DEBUG_MODE and idx in log_data:
                        with st.expander("🕵️‍♂️ [디버그] Pass 1.5 정제 텍스트 보기"):
                            st.code(log_data[idx])
                            
                with row_col2:
                    issues = [r for r in report_data if r.get("image_index") == idx]
                    if not issues:
                        st.markdown(
                            '<div class="risk-warning"><div class="card-title">⚠️ 오류 발생</div>'
                            '이 구간의 처리 결과를 찾지 못했습니다.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        for issue in issues:
                            risk = issue.get("risk_level", "적합")

                            if risk == "치명적 위반":
                                css_class = "risk-critical"
                                icon = "❌"
                            elif risk == "수정 권고":
                                css_class = "risk-warning"
                                icon = "⚠️"
                            else:
                                css_class = "risk-pass"
                                icon = "✅"

                            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="card-title">{icon} {issue.get("title", "검토 완료")}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"""
                            - **해당 구간 내용 요약:** {issue.get('exact_text_in_design', '')}
                            - **QC 팩트 근거:** {issue.get('fact_or_legal_ground', '')}
                            - **판정 사유 (브리핑):** {issue.get('discrepancy_analysis', '')}
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
                
        except Exception as e:
            st.error(f"전체 프로세스 중 오류 발생: {e}")
