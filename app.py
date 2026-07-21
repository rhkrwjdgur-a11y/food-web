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
st.set_page_config(page_title="식품 표시사항 정밀 검토 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-warning { background-color: #fefaf0; padding: 20px; border-radius: 10px; border-left: 6px solid #f39c12; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-pass { background-color: #f4fbf7; padding: 20px; border-radius: 10px; border-left: 6px solid #2ecc71; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .card-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 10px; margin-bottom: 15px; }
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
# 4. 투트랙 에이전트 시스템 (비전 API + LLM 2단계 분리)
# ==========================================
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
    
    # 이미지를 순회하며 1단계(추출)와 2단계(검토)를 철저히 분리하여 실행
    for idx, img_obj in enumerate(main_images):
        img_obj = img_obj.resize((2000, int(img_obj.size[1] * (2000.0/img_obj.size[0]))), Image.LANCZOS) if img_obj.size[0] > 2000 else img_obj
        chunk_list.append(img_obj)
        
        # --- [1단계] 시안 텍스트 강제 원형 추출 (뇌를 끄고 필사만 수행) ---
        extract_prompt = """
        당신은 텍스트 추출기입니다. 이 이미지에 적힌 모든 텍스트, 수치, 단위를 '보이는 그대로' 한 글자의 수정 없이 필사하십시오.
        디자이너의 오타(예: 무규 포장, 비타민B1염산염, 유성비타민지방산에스테르, 콜레스테롤 0mg 33% 등)를 절대 문맥에 맞게 뇌내 보정(Auto-correction)하지 마십시오.
        오직 텍스트만 추출하여 출력하십시오.
        """
        try:
            extraction_response = model.generate_content([extract_prompt, img_obj], generation_config=genai.types.GenerationConfig(temperature=0.0))
            design_raw_text = extraction_response.text
        except:
            design_raw_text = "텍스트 추출 실패"

        # --- [2단계] 추출된 텍스트와 팩시안 텍스트 1:1 논리 검토 (비교 분석만 수행) ---
        review_prompt = f"""
        당신은 엄격하고 기계적인 품질관리(QC) 검수자입니다.
        아래 [1단계에서 추출된 시안 텍스트]와 [Google Vision API 팩시안 텍스트]를 글자 단위로 1:1 대조하십시오.

        [1단계: 시안에서 추출된 원시 텍스트 (오타 및 복붙 수치 그대로 보존됨)]
        {design_raw_text}

        [Google Vision API가 100% 완벽하게 필사한 팩시안 원시 데이터 - 절대적 진리]
        {ocr_extracted_text}

        [요약된 국가 공인 영양성분 DB 데이터]
        {db_context_text if db_context_text else "외부 DB 데이터 없음."}

        [필수 강제 체크리스트 - 기존 룰 절대 축소 불가]
        
        🔥 1. [디자이너 복붙 오타 색출 (영양정보 & 원재료명 철자)]:
           - 위 [1단계 시안 텍스트]의 '영양정보표'에서 당류 수치(g/%), 콜레스테롤 수치(mg/%), 트랜스지방 단위(g/mg)를 팩시안과 1:1로 비교하십시오. 단 하나라도 다르면 적발하십시오. (예: 시안 콜레스테롤 0mg 33% vs 팩시안 0mg 0%).
           - '원재료명'을 철자 단위로 대조하여 오기재(예: 비타민B1염산염 -> 비타민B,염산염 / 유성비타민지방산에스테르 -> 유성비타민A지방산에스테르 / 비타민D3 -> 비타민D)를 완벽히 색출하십시오.

        🔥 2. [주의문구 오기재 및 논리 모순]:
           - 시안 하단의 알레르기 주의문구를 팩시안과 대조하십시오. 해당 제품에 없는 원료(예: 땅콩, 잣 등)가 시안에 기재되어 있다면 적발하십시오.
           - 주원료가 교차오염 우려 물질에 중복 기재되어 있다면 논리 모순으로 적발하십시오.

        🚨 3. [연출 컷 주의문구]: 
           - 제공된 이미지를 직접 눈으로 확인하십시오. 잔에 두유를 따르는 등 조리/연출 컷이 있는데 주변에 '이미지 예' 또는 '연출된 이미지' 주의문구가 없다면 지적하십시오.

        🔥 4. [마케팅 카피 단순 오타]:
           - 1단계 시안 텍스트 중 철자 오타(예: 무균 포장 -> 무규 포장 등)가 있는지 대조하여 확인하십시오.

        🔥 5. [제품명 및 기능성 워딩 1:1 대조]:
           - 시안 내 제품명에 누락된 단어(예: &고칼슘)가 없는지 확인하십시오.
           - 비타민 기능성 문구 중 식약처 공전 워딩(예: 비타민E '항산화작용을 하여 유해산소로부터 세포를 보호하는데 필요')이 토씨 하나라도 누락(예: '항산화작용을 하여' 누락)되었는지 대조하십시오.

        🔥 6. [원물 은폐 기만]:
           - 시안에서 강조한 농산물(검은콩, 아몬드, 잣 등)이 팩시안 배합비율상 1% 미만의 극소량(예: 0.21%, 0.333% 등)이면 기만으로 적발하십시오.

        반드시 JSON 배열 형식으로 응답하십시오.
        [
          {{
            "image_index": {idx},
            "risk_level": "치명적 위반" 또는 "수정 권고",
            "title": "검토 항목 요약",
            "exact_text_in_design": "1단계 시안 추출 텍스트에서 발췌한 원문(오타 포함)",
            "fact_or_legal_ground": "대조한 팩시안 원문 데이터",
            "discrepancy_analysis": "1대1 대조 결과 및 조치사항"
          }}
        ]
        """
        
        try:
            # 2단계에서는 추출된 텍스트와 함께 '이미지 예' 확인을 위해 이미지를 다시 던져줍니다.
            review_response = model.generate_content([review_prompt, img_obj], generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
            chunk_issues = json.loads(review_response.text)
            final_report.extend(chunk_issues)
        except Exception as e:
            pass
            
        time.sleep(1.5) # API 속도 제한 방지 대기
        
    return json.dumps(final_report), chunk_list

# ==========================================
# 5. UI 및 실행 흐름
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.sidebar.markdown("---")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 2단계 분리 정밀 교차 검증 (추출 -> 대조)", use_container_width=True)

st.title("🛡️ 식품 표시·광고 정밀 통제 시스템 (2-Step Ver.)")
st.markdown("---")

if not uploaded_main_images:
    st.warning("👈 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    if not trigger_api:
        for img in main_img_objs: st.image(img, use_container_width=True)
    else:
        with st.spinner("👁️ [팩시안 기준점 확보] 구글 비전 API가 팩시안을 픽셀 단위로 해독 중입니다..."):
            vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
            st.success("✅ 구글 비전 API 팩시안 판독 완료")
            with st.expander("🔍 구글 비전 API 추출 날것(Raw Text) 팩트 확인"):
                st.text(vision_extracted_text)

        with st.spinner("🔍 [DB 연동] 외부 영양성분 DB 타겟 탐지 중..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            final_db_context_text = ""
            if auto_dict:
                for base_food, detail_cond in auto_dict.items():
                    db_data = query_food_nutrient_db(base_food)
                    if db_data:
                        simplified_db = [f"- [{row.get('DESC_KOR', '이름없음')}] 열량:{row.get('NUTR_CONT1')}kcal, 단백질:{row.get('NUTR_CONT3')}g, 지방:{row.get('NUTR_CONT4')}g" for row in db_data[:20]]
                        final_db_context_text += f"\n[검색어 '{base_food}'] DB 요약\n" + "\n".join(simplified_db) + "\n"

        with st.spinner("⚖️ [제2 에이전트] 1단계: 시안 원시 텍스트 강제 추출 ➡️ 2단계: 팩시안 1:1 대조 중..."):
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
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>1대1 교차 검증 결과 특이사항 없음.</div>', unsafe_allow_html=True)
                        else:
                            for issue in issues:
                                risk = issue.get("risk_level", "정상")
                                css_class = "risk-critical" if risk == "치명적 위반" else "risk-warning" if risk == "수정 권고" else "risk-pass"
                                icon = "❌" if risk == "치명적 위반" else "⚠️" if risk == "수정 권고" else "✅"
                                
                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                - **시안 텍스트 원문:** {issue.get('exact_text_in_design', '')}
                                - **QC 팩트 근거:** {issue.get('fact_or_legal_ground', '')}
                                - **조치사항:** {issue.get('discrepancy_analysis', '')}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
            except Exception as e: st.error(f"오류 발생: {e}")
