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
    # 1) 제미나이 API 연동
    genai.configure(api_key=st.secrets["AI_VISION_API_KEY"])
    
    # 2) 식약처 DB API 키 연동
    FOOD_API_KEY = st.secrets["FOOD_SAFETY_API_KEY"]
    
    # 3) 구글 클라우드 비전 API 연동 (JSON 문자열 파싱 방식)
    gcp_json_string = st.secrets["gcp_service_account"]["GOOGLE_VISION_KEY"]
    gcp_credentials = json.loads(gcp_json_string)
    
    # 파싱 과정에서 이스케이프된 줄바꿈 기호를 실제 줄바꿈 문자로 완벽히 복구
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
# 3. 보조 함수 (법령 가이드라인, 식약처 DB 조회)
# ==========================================
@st.cache_data
def load_guideline_knowledge():
    docs_path = "docs"
    knowledge_text = ""
    if not os.path.exists(docs_path): return "", "문서 폴더 없음"
    for filename in [f for f in os.listdir(docs_path) if f.endswith('.pdf')]:
        try:
            with open(os.path.join(docs_path, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i in range(min(len(reader.pages), 20)):
                    text = reader.pages[i].extract_text()
                    if text: knowledge_text += text + "\n"
        except: pass
    return knowledge_text, None

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
    payload.append("""타 식품과 수치를 비교하거나, 식약처 등 공식 DB를 인용하는 문구가 있다면 대분류 명사를 key로, 조건을 value로 JSON 출력. 없으면 "NONE" 출력.""")
    try:
        res = model.generate_content(payload, generation_config=genai.types.GenerationConfig(temperature=0.0)).text.strip()
        if res == "NONE": return {}
        return json.loads(re.sub(r'```json\s*|```\s*', '', res))
    except: return {}

# ==========================================
# 4. 투트랙 에이전트 시스템 (비전 API + LLM)
# ==========================================
def extract_text_with_google_vision(uploaded_files):
    """제1 에이전트: 구글 비전 API로 팩시안 밀집 텍스트 무결점 추출"""
    if not uploaded_files:
        return "팩시안 이미지 없음"
    
    extracted_text = ""
    for file in uploaded_files:
        content = file.read()
        image = vision.Image(content=content)
        # 밀집 텍스트(영수증, 문서 등) 판독 모드 적용
        response = vision_client.document_text_detection(image=image)
        
        if response.error.message:
            st.error(f"Vision API 에러: {response.error.message}")
            continue
            
        if response.full_text_annotation:
            extracted_text += response.full_text_annotation.text + "\n\n"
            
        file.seek(0)
        
    return extracted_text

def analyze_design_with_ai(main_images, ocr_extracted_text, db_context_text):
    """제2 에이전트: 제미나이 LLM으로 1대1 팩트 교차 검증 수행"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    content_payload = []
    chunk_list = []
    
    for idx, img_obj in enumerate(main_images):
        img_obj = img_obj.resize((2000, int(img_obj.size[1] * (2000.0/img_obj.size[0]))), Image.LANCZOS) if img_obj.size[0] > 2000 else img_obj
        chunk_list.append(img_obj)
        content_payload.extend([f"--- [시안 구간 인덱스: {idx}] ---", img_obj])
                
    prompt = f"""
    당신은 엄격하고 기계적인 품질관리(QC) 검수자입니다. 당신의 가장 중요한 임무는 '마케팅 시안 이미지'에 표기된 모든 수치, 텍스트가 구글 비전 API가 추출해 준 [확정된 팩시안 원시 데이터]와 1대1로 완벽하게 일치하는지 '글자 단위로 교차 검증'하는 것입니다.
    
    [Google Vision API가 100% 완벽하게 필사한 팩시안(후면라벨) 원시 데이터 - 절대적 진리]
    {ocr_extracted_text}
    
    [요약된 국가 공인 영양성분 DB 데이터]
    {db_context_text if db_context_text else "외부 DB 데이터 없음."}
    
    [필수 강제 체크리스트 - 스킵 절대 금지]
    
    🔥 1. [1대1 제품명 및 영양정보 대조 (가장 중요)]:
       - 시안에 적힌 '제품명(예: 연세두유 고단백 검은콩)'이 팩시안의 정식 제품명(예: 연세두유 고단백 검은콩&고칼슘)과 토씨 하나라도 다르거나 누락된 단어가 있다면 "수정 권고"하십시오.
       - 시안의 '영양정보' 표기(열량, 당류, 콜레스테롤, 트랜스지방 등)를 팩시안과 단위(g/mg), 수치, 비율(%)까지 1대1로 대조하십시오. (예: 시안 '당류 1g 1%' vs 팩시안 '당류 4g 4%' / 시안 '콜레스테롤 0mg 33%' vs 팩시안 '0mg 0%' / 시안 '트랜스지방 0mg' vs 팩시안 '0g'). 단 하나라도 다르면 "치명적 위반(오기재)"으로 적발하십시오.

    🔥 2. [알레르기 주의문구 1대1 검증 및 논리 모순]:
       - 시안의 알레르기 주의문구를 팩시안의 주의문구와 1대1로 대조하여 없는 원료(예: 땅콩, 잣 등)가 들어갔거나 누락되었는지 확인하십시오.
       - 특히 제품의 주원료(예: 잣, 아몬드 등)가 주의문구(교차오염 우려)에 중복으로 기재되어 있다면 논리 모순으로 적발하십시오.

    🚨 3. [연출 컷 주의문구]: 잔에 따르는 모습 등 조리/연출 이미지 주변에 '연출된 이미지' 또는 '이미지 예' 주의문구가 누락되었다면 지적하십시오.

    🔥 4. [기능성 명칭 1대1 검증]: 비타민 등 영양소 기능성 설명 시 식약처 공전 워딩(예: 비타민E '항산화작용을 하여 유해산소로부터 세포를 보호')과 1대1로 대조하여 '항산화작용을 하여' 등이 누락되었다면 정확히 수정 권고하십시오.

    🔥 5. [의무표시와 기만표시 구분]: 영양정보표 내 '콜레스테롤 0mg' 표기는 합법입니다. 마케팅 카피에서 '콜레스테롤 NO' 등을 강조했다면 기만으로 적발하십시오.

    🔥 6. [원물 은폐 기만]: 시안에서 강조한 농산물(검은콩, 아몬드, 잣 등)이 팩시안 배합비율상 1% 미만의 극소량(예: 0.21%, 0.333% 등)이면 주원료 기만으로 "치명적 위반" 적발하십시오.
    
    반드시 JSON 배열 형식으로 응답하십시오. 분석을 수행할 때, discrepancy_analysis 항목 내에 무조건 "1대1 대조 결과: [시안 표기] vs [팩시안 표기]" 문장을 포함하여 양쪽 데이터를 철저히 비교했음을 증명하십시오.
    [
      {{
        "image_index": 구간 인덱스 번호,
        "risk_level": "치명적 위반" 또는 "수정 권고" 또는 "정상",
        "title": "검토 항목 요약",
        "marketing_text": "상세페이지 마케팅/영양 표기 원문",
        "fact_or_legal_ground": "대조한 Google Vision API 팩트 또는 식약처 규정",
        "discrepancy_analysis": "1대1 대조 결과: (시안 내용 vs 팩시안 내용) / 위반 분석 및 조치 사항"
      }}
    ]
    """
    content_payload.append(prompt)
    
    for attempt in range(3):
        try:
            response = model.generate_content(content_payload, generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
            return response.text, chunk_list
        except Exception as e:
            if attempt < 2: time.sleep(10); continue
            raise e

# ==========================================
# 5. UI 및 실행 흐름
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.sidebar.markdown("---")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 1대1 교차 검증 (Vision API + LLM)", use_container_width=True)

st.title("🛡️ 식품 표시·광고 정밀 통제 시스템 (Cross-check Ver.)")
st.markdown("---")

if not uploaded_main_images:
    st.warning("👈 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    if not trigger_api:
        for img in main_img_objs: st.image(img, use_container_width=True)
    else:
        # [Step 1] Google Cloud Vision API 텍스트 추출
        with st.spinner("👁️ [제1 에이전트] 구글 비전 API가 팩시안을 픽셀 단위로 해독 중입니다..."):
            vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
            st.success("✅ 구글 비전 API 판독 완료")
            with st.expander("🔍 구글 비전 API 추출 날것(Raw Text) 팩트 확인"):
                st.text(vision_extracted_text)

        # [Step 2] 식약처 DB 연동
        with st.spinner("🔍 [DB 연동] 외부 영양성분 DB 타겟 탐지 중..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            final_db_context_text = ""
            if auto_dict:
                for base_food, detail_cond in auto_dict.items():
                    db_data = query_food_nutrient_db(base_food)
                    if db_data:
                        simplified_db = [f"- [{row.get('DESC_KOR', '이름없음')}] 열량:{row.get('NUTR_CONT1')}kcal, 단백질:{row.get('NUTR_CONT3')}g, 지방:{row.get('NUTR_CONT4')}g" for row in db_data[:20]]
                        final_db_context_text += f"\n[검색어 '{base_food}'] DB 요약\n" + "\n".join(simplified_db) + "\n"

        # [Step 3] 제미나이 LLM 1:1 대조 분석
        with st.spinner("⚖️ [제2 에이전트] 제미나이가 팩시안과 시안을 1:1로 정밀 교차 검증 중입니다..."):
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
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>1대1 교차 검증 및 법적 테두리 확인 완료.</div>', unsafe_allow_html=True)
                        else:
                            for issue in issues:
                                risk = issue.get("risk_level", "정상")
                                css_class = "risk-critical" if risk == "치명적 위반" else "risk-warning" if risk == "수정 권고" else "risk-pass"
                                icon = "❌" if risk == "치명적 위반" else "⚠️" if risk == "수정 권고" else "✅"
                                
                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                - **마케팅 원문:** {issue.get('marketing_text', '')}
                                - **QC 팩트 근거:** {issue.get('fact_or_legal_ground', '')}
                                - **조치사항:** {issue.get('discrepancy_analysis', '')}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
            except Exception as e: st.error(f"오류 발생: {e}")
