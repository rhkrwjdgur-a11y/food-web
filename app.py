import streamlit as st
import google.generativeai as genai
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import os
import PyPDF2
import json
import time
import requests
import urllib.parse
import re

# ==========================================
# 1. 기본 페이지 설정 및 CSS
# ==========================================
st.set_page_config(page_title="상세페이지 정밀 검토 시스템", layout="wide")

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
# 2. 3대 핵심 API 키 연동 (Secrets)
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
except json.JSONDecodeError as e:
    st.error(f"구글 비전 JSON 키 형식이 잘못되었습니다: {e}")
    st.stop()
except Exception as e:
    st.error(f"구글 비전 인증 오류: {e}")
    st.stop()

# ==========================================
# 3. 보조 함수 
# ==========================================
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
    model = genai.GenerativeModel('gemini-2.5-flash')
    payload = [img.resize((1000, int(img.size[1] * (1000.0/img.size[0]))), Image.LANCZOS) if img.size[0] > 1000 else img for img in main_images]
    payload.append("""타 식품과 수치를 비교하거나 식약처 DB를 인용하는 문구가 있다면 대분류 명사를 key로 JSON 출력. 없으면 "NONE" 출력.""")
    try:
        res = model.generate_content(payload, generation_config=genai.types.GenerationConfig(temperature=0.0)).text.strip()
        if res == "NONE": return {}
        return json.loads(re.sub(r'```json\s*|```\s*', '', res))
    except: return {}

# ==========================================
# 4. 투트랙 에이전트 시스템 (캐싱 및 환각 차단 고도화)
# ==========================================
@st.cache_data(show_spinner=False)
def extract_text_with_google_vision(uploaded_files):
    if not uploaded_files: return "팩시안 이미지 없음"
    extracted_text = ""
    for file in uploaded_files:
        content = file.read()
        image = vision.Image(content=content)
        response = vision_client.document_text_detection(image=image)
        if response.error.message:
            st.error(f"Vision API 에러: {response.error.message}")
            continue
        if response.full_text_annotation:
            extracted_text += response.full_text_annotation.text + "\n\n"
        file.seek(0)
    return extracted_text

def analyze_design_with_ai(main_images, ocr_extracted_text, db_context_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    final_report = []
    chunk_list = []
    
    for idx, img_obj in enumerate(main_images):
        img_obj = img_obj.resize((2000, int(img_obj.size[1] * (2000.0/img_obj.size[0]))), Image.LANCZOS) if img_obj.size[0] > 2000 else img_obj
        chunk_list.append(img_obj)
        
        # --- [1단계] 시안 텍스트 강제 원형 추출 (환각 차단 1) ---
        extract_prompt = """
        당신은 시력 2.0의 철저하고 객관적인 텍스트 추출기입니다. 
        현재 보여지는 '이 하나의 이미지 구간'에 실제로 적혀있는 텍스트와 숫자만 픽셀 단위로 필사하십시오.
        [치명적 오류 주의]: 이미지에 없는 텍스트(예: 이전 이미지 구간에서 본 내용이나, 팩시안 등 다른 곳에 있는 내용)를 상상해서 덧붙이거나 지어내면 절대 안 됩니다.
        텍스트가 아예 없다면 "텍스트 없음"이라고만 출력하십시오.
        디자이너의 오타를 임의로 고치지 말고 날것 그대로 추출하십시오.
        """
        try:
            extraction_response = model.generate_content([extract_prompt, img_obj], generation_config=genai.types.GenerationConfig(temperature=0.0))
            design_raw_text = extraction_response.text
        except:
            design_raw_text = "텍스트 추출 실패"

        # --- [2단계] 선임자급 핀셋 검수 (환각 차단 2 및 과잉 지적 금지) ---
        review_prompt = f"""
        당신은 실무 경험이 풍부한 식품 마케팅 상세페이지 QC 선임자입니다. 
        아래 [1단계 시안 텍스트]와 [Google Vision API 팩시안 데이터]를 대조하십시오.

        [1단계 시안 텍스트 (현재 이미지 구간에 실제로 존재하는 글자)]
        {design_raw_text}

        [Google Vision API 팩시안 원시 데이터 - 절대적 기준]
        {ocr_extracted_text}

        🚨 [치명적 환각(Hallucination) 및 과잉 지적 금지 절대 규칙] 🚨
        1. **없는 내용 창조 지적 절대 금지:** 위 [1단계 시안 텍스트]에 명확히 존재하지 않는 단어(예: 나이아신, 단백질 12g, 특정 성분 수치 등)를 당신이 팩시안만 보고 임의로 상상해서 "시안에 오기재되었다", "불일치한다"고 지적하는 것은 최악의 시스템 오류입니다. 오직 [1단계 시안 텍스트]에 적힌 텍스트 안에서만 검수하십시오.
        2. **해당 구간에 없으면 패스:** 현재 이미지 구간에 영양정보표나 특정 원재료명이 없다면, "누락되었다"고 지적하지 말고 팩트 체크 대상이 아니므로 무조건 "정상"으로 처리하여 통과하십시오.

        🔥 [상세페이지 핵심 핀셋 검수 룰 (반드시 확인)]
        1. [영양정보 복붙 오타 색출]: [1단계 시안 텍스트]에 '영양정보'와 관련된 수치가 적혀있을 때만 팩시안과 1:1 대조하십시오. (당류 1g 1%, 콜레스테롤 0mg 33% 등 복붙 오류 적발)
        2. [알레르기 문구 대조]: [1단계 시안 텍스트]에 '주의문구'가 있을 때만 팩시안에 없는 원료(땅콩, 잣 등)가 잘못 포함되었는지 대조하십시오.
        3. [제품명 오기재]: 카피나 제품명에 '연세두유 고단백 검은콩'이라고만 되어 있다면 '&고칼슘'이 누락된 것이므로 추가하라고 지적하십시오.
        4. [비타민 기능성 명칭]: 비타민E 설명 텍스트가 있을 때, '항산화작용을 하여'가 빠져있다면 지적하십시오.
        5. [연출 컷 주의문구]: 제공된 이미지를 직접 보고, 잔에 두유를 따르거나 원물(고기, 생선, 콩, 잣 등)이 있는 이미지 구간에는 반드시 '이미지 예' 또는 '연출된 이미지'라는 문구가 있어야 합니다. 이미지가 연출 컷인데 주의문구가 없으면 지적하십시오.
        6. [카피 오타]: 마케팅 카피 중 '무규 포장'이라는 오타가 있으면 '무균 포장'으로 고치라고 지적하십시오.
        7. [원재료명 오기재]: [1단계 시안 텍스트]에 원재료명이 적혀있을 때만 철자 오류(예: 유성비타민지방산에스테르)를 찾아내십시오.
        8. [원물 은폐 기만]: 시안에서 검은콩이나 아몬드&잣을 크게 강조하는데, 팩시안 배합비율이 1% 미만의 극소량(예: 0.21%)이라면 기만 소지가 있음을 지적하십시오.

        반드시 JSON 배열 형식으로 응답하십시오.
        [
          {{
            "image_index": {idx},
            "risk_level": "치명적 위반" 또는 "수정 권고" 또는 "정상",
            "title": "검토 항목 요약",
            "exact_text_in_design": "1단계 시안 텍스트에서 발췌한 실제 텍스트 (또는 해당 이미지 연출 컷 설명)",
            "fact_or_legal_ground": "팩시안 원문 데이터",
            "discrepancy_analysis": "위반 사항에 대한 명확한 지적 내용"
          }}
        ]
        """
        
        try:
            review_response = model.generate_content([review_prompt, img_obj], generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
            chunk_issues = json.loads(review_response.text)
            filtered_issues = [issue for issue in chunk_issues if issue.get("risk_level") != "정상"]
            final_report.extend(filtered_issues)
        except Exception as e:
            pass
            
        time.sleep(1.5) 
        
    return json.dumps(final_report), chunk_list

# ==========================================
# 5. UI 및 실행 흐름
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.sidebar.markdown("---")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 상세페이지 핀셋 교차 검증", use_container_width=True)

st.title("🛡️ 마케팅 상세페이지 정밀 통제 시스템 (Sniper Ver.)")
st.markdown("---")

if not uploaded_main_images:
    st.warning("👈 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    if not trigger_api:
        for img in main_img_objs: st.image(img, use_container_width=True)
    else:
        with st.spinner("👁️ [팩시안 기준점 확보] 구글 비전 API가 팩시안을 해독 중입니다... (캐시 적용)"):
            vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
            st.success("✅ 구글 비전 API 팩시안 판독 완료")

        with st.spinner("🔍 [DB 연동] 외부 영양성분 DB 타겟 탐지 중..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            final_db_context_text = ""
            if auto_dict:
                for base_food, detail_cond in auto_dict.items():
                    db_data = query_food_nutrient_db(base_food)
                    if db_data:
                        simplified_db = [f"- [{row.get('DESC_KOR', '이름없음')}] 열량:{row.get('NUTR_CONT1')}kcal, 단백질:{row.get('NUTR_CONT3')}g, 지방:{row.get('NUTR_CONT4')}g" for row in db_data[:20]]
                        final_db_context_text += f"\n[검색어 '{base_food}'] DB 요약\n" + "\n".join(simplified_db) + "\n"

        with st.spinner("⚖️ [제2 에이전트] 환각을 차단하고 텍스트에 실재하는 오류만 핀셋으로 색출합니다..."):
            try:
                json_result, chunk_list = analyze_design_with_ai(main_img_objs, vision_extracted_text, final_db_context_text)
                report_data = json.loads(json_result)
                
                for idx, chunk_img in enumerate(chunk_list):
                    st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                    row_col1, row_col2 = st.columns([1, 1])
                    with row_col1: st.image(chunk_img, use_container_width=True)
                    with row_col2:
                        issues = [r for r in report_data if r.get("image_index") == idx]
                        if not issues: 
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>이 구간에는 팩시안과 불일치하는 오기재나 수정 사항이 발견되지 않았습니다.</div>', unsafe_allow_html=True)
                        else:
                            for issue in issues:
                                risk = issue.get("risk_level", "정상")
                                css_class = "risk-critical" if risk == "치명적 위반" else "risk-warning"
                                icon = "❌" if risk == "치명적 위반" else "⚠️"
                                
                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                - **시안 텍스트/이미지:** {issue.get('exact_text_in_design', '')}
                                - **QC 팩트 기준:** {issue.get('fact_or_legal_ground', '')}
                                - **조치사항:** {issue.get('discrepancy_analysis', '')}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
            except Exception as e: st.error(f"오류 발생: {e}")
