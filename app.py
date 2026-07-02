import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2
import json
import time
import requests
import urllib.parse
from datetime import datetime

# 1. 기본 페이지 설정
st.set_page_config(page_title="식품 표시사항 정밀 검토 시스템", layout="wide")

# CSS 디자인
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-warning { background-color: #fefaf0; padding: 20px; border-radius: 10px; border-left: 6px solid #f39c12; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-pass { background-color: #f4fbf7; padding: 20px; border-radius: 10px; border-left: 6px solid #2ecc71; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .card-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 10px; margin-bottom: 15px; }
    .metric-box { text-align: center; background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
    .metric-num { font-size: 26px; font-weight: bold; margin-top: 5px; display: block; }
    </style>
    """, unsafe_allow_html=True)

# 2. API 키 설정
try:
    genai.configure(api_key=st.secrets["AI_VISION_API_KEY"])
    FOOD_API_KEY = st.secrets["FOOD_SAFETY_API_KEY"]
except KeyError as e:
    st.error(f"시스템 오류: Secrets 설정 누락 - {e}")

# 3-1. 법령 가이드라인 PDF 로드 함수
@st.cache_data
def load_guideline_knowledge():
    docs_path = "docs"
    knowledge_text = ""
    if not os.path.exists(docs_path): return "", "문서 폴더 없음"
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    if not pdf_files: return "", "PDF 없음"
    for filename in pdf_files:
        try:
            with open(os.path.join(docs_path, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i in range(min(len(reader.pages), 20)):
                    text = reader.pages[i].extract_text()
                    if text: knowledge_text += text + "\n"
        except Exception: pass
    return knowledge_text, None

# 3-2. 식약처 영양성분 DB 호출 함수
def query_food_nutrient_db(food_name):
    if not food_name: return None
    service_id = "I2790"
    encoded_food = urllib.parse.quote(food_name.strip())
    url = f"http://openapi.foodsafetykorea.go.kr/api/{FOOD_API_KEY}/{service_id}/json/1/50/DESC_KOR={encoded_food}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if service_id in res_json and 'row' in res_json[service_id]:
                return res_json[service_id]['row']
    except Exception: pass
    return None

# 4-1. Auto Pre-Scan: 이미지에서 비교 대상 식품 키워드 자동 추출 (범용 룰 적용)
def auto_extract_db_keywords(main_images):
    model = genai.GenerativeModel('gemini-2.5-flash')
    payload = []
    for img in main_images:
        w, h = img.size
        if w > 1000: img = img.resize((1000, int(h * (1000.0/w))), Image.LANCZOS)
        payload.append(img)
    prompt = """
    당신은 식품 상세페이지에서 외부 영양성분 DB 검색이 필요한 '핵심 원물 명칭'을 자동 추출하는 AI입니다.
    이미지들을 훑어보고, 자사 제품과 타 식품군의 영양성분 수치를 비교하는 마케팅 자료(표, 그래프 등)가 있는지 찾으십시오.
    발견되었다면, 수식어(구운것, 말린것, 생것 등)를 제외한 가장 기본이 되는 식품의 명사만을 쉼표(,)로 구분하여 출력하십시오. (예: 쇠고기, 닭고기, 대두, 우유, 사과)
    절대 다른 부연 설명 없이 명사들만 출력하십시오. 없다면 오직 'NONE'이라고만 출력하십시오.
    """
    payload.append(prompt)
    try:
        response = model.generate_content(payload)
        res_text = response.text.strip()
        if res_text == "NONE" or not res_text: return []
        return [k.strip() for k in res_text.split(",")]
    except Exception: return []

# 4-2. 실시간 AI 비전 분석 로직 (특정 단어 저격 금지, 범용 논리 탑재)
def analyze_design_with_ai(main_images, ref_files, master_fact_files, legal_text, db_context_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    current_date_str = datetime.now().strftime("%Y년 %m월 %d일")
    
    content_payload = []
    chunk_list = []
    
    for img_obj in main_images:
        w, h = img_obj.size
        if w > 2000: 
            img_obj = img_obj.resize((2000, int(h * (2000.0/w))), Image.LANCZOS)
        chunk_list.append(img_obj)
        content_payload.append(img_obj)
            
    if ref_files:
        for ref in ref_files:
            try: content_payload.append(Image.open(ref))
            except: pass 
    master_fact_count = 0
    if master_fact_files:
        for fact in master_fact_files:
            try: content_payload.append(Image.open(fact)); master_fact_count += 1
            except: pass
                
    prompt = f"""
    당신은 엄격한 품질관리(QC) 전문가입니다. 특정 제품에 국한되지 않는 범용적인 잣대로 각 시안 조각(인덱스)을 검토하십시오.
    
    [식약처 법령 지식 베이스]
    {legal_text}
    
    [자동 추출된 국가 공인 영양성분 DB 데이터]
    {db_context_text if db_context_text else "경고: 외부 DB 데이터가 존재하지 않습니다. 수치 비교가 불가함을 리포트에 명시하십시오."}
    
    [필수 강제 체크리스트 - 범용성 보장]
    🔥 1. DB 비교 수치 범용 검증:
       - 시안 내 타 식품 비교 자료에 적힌 원물 명칭과 세부 수식어(부위, 조리법 등)를 모두 읽으십시오.
       - 위 [식약처 DB 데이터]와 부위/조리 상태(DESC_KOR)가 일치하는 항목의 수치를 대조하여, 일치 시 "risk_level": "정상" 객체를 생성하여 "식약처 DB 수치와 일치하여 적합함"을 명시하십시오. 불일치 시 "치명적 위반"으로 적발하십시오.

    🔥 2. 시간 조작 과장광고 범용 방어:
       - 매출 1위, 수상 내역 등의 데이터 산정 기간이 명시된 경우, 해당 기간이 현재 시점({current_date_str})을 초과하는 미래 시점의 데이터를 근거로 삼고 있는지 팩트 대조하여 과장 광고를 적발하십시오.

    🔥 3. 배합 기만 및 모순 범용 방어:
       - 마케팅 문구나 제조 공정도 상에 특정 하위 원료(예: 특정 품종, 특정 부위)를 '100%'로 강조하거나 단독 사용한 것처럼 묘사했을 때, 실제 원재료명에 상위 범주의 범용 원료나 타 원료가 혼합되어 있는지 대조하십시오.
       - 상하단 칼로리 모순 에러는 '영양정보표'가 명확히 보이는 해당 인덱스에서만 1번 출력하십시오.
       
    🔥 4. 영양강조표시 및 연출 사진 범용 기준:
       - 원재료 유래 천연 당류로 인해 영양정보표 상 당류가 0g을 초과함에도 마케팅 시안에 'ZERO' 등의 절대적 표현을 사용했는지 대조하여 '무첨가'로의 수정을 권고하십시오.
       - 조리되거나 원물이 묘사된 연출 사진 주변에 '연출된 이미지' 등의 면책 문구가 누락되었는지 확인하십시오.
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오.
    [
      {{
        "image_index": 구간 인덱스 번호 (0부터 시작, 업로드된 이미지 순서와 동일함),
        "risk_level": "치명적 위반" 또는 "수정 권고" 또는 "정상",
        "title": "검토 항목 요약",
        "marketing_text": "상세페이지 추출 원문",
        "fact_or_legal_ground": "팩시안, 식약처 DB 매칭 항목, 날짜 팩트 또는 법적 가이드라인",
        "discrepancy_analysis": "위반 분석 및 조치 사항"
      }}
    ]
    * 해당 이미지에 위반사항이나 DB 적합 판정이 전혀 없다면 risk_level "정상" 객체(내용: 특이사항 없음)를 반환하십시오.
    """
    content_payload.append(prompt)
    
    for attempt in range(3):
        try:
            response = model.generate_content(content_payload, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
            return response.text, chunk_list
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                if attempt < 2: time.sleep(10); continue
            raise e

# ==========================================
# 왼쪽 사이드바
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 식약처 영양성분 DB 실시간 자동 연동 (비교광고 검증용)")
db_search_keyword = st.sidebar.text_input("상세페이지 내 비교 대상 식품명 입력", help="비워두면 AI가 자동으로 탐지합니다.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서 및 추가 근거 자료", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 3-Pass 투트랙 + 식약처 DB 자동 정밀 심사", use_container_width=True)

# ==========================================
# 최상단
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 식약처 부당광고 고시 및 영양표시 지침 실시간 학습 완료")

# ==========================================
# 메인 화면
# ==========================================
if not uploaded_main_images:
    st.warning("👈 왼쪽 메뉴에서 의미 단위(배경/흐름)로 캡처하신 상세페이지 시안 이미지를 순서대로 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
        
    if not trigger_api:
        st.info("좌측 하단의 심사 가동 버튼을 누르면 AI 분석이 시작됩니다.")
        for img in main_img_objs: st.image(img, use_container_width=True)
    else:
        final_db_context_text = "" 
        
        with st.spinner("🔍 1단계: 시안 내 식약처 DB 비교광고 키워드를 자동 탐지하고 있습니다..."):
            auto_keywords = auto_extract_db_keywords(main_img_objs)
            if auto_keywords:
                st.sidebar.success(f"🤖 AI 자동 탐지 핵심 키워드: {', '.join(auto_keywords)}")
                for kw in auto_keywords:
                    db_data = query_food_nutrient_db(kw)
                    if db_data:
                        final_db_context_text += f"\n[검색어 '{kw}' 식약처 공인 데이터 (최대 50건)]\n" + json.dumps(db_data[:50], ensure_ascii=False) + "\n"
                        st.sidebar.info(f"✅ 식약처 DB '{kw}' 연동 완료 (상세 분류 포함)")
                    else:
                        st.sidebar.error(f"❌ DB에서 '{kw}' 데이터를 찾을 수 없습니다.")
            else:
                st.sidebar.info("🔍 탐지된 비교광고 외부 DB 키워드 없음")

        with st.spinner("⚙️ 2단계: 3-Pass 투트랙 정밀 심사 가동 중 (범용 QC 룰 적용)..."):
            try:
                ref_files = []
                if uploaded_test: ref_files.extend(uploaded_test)
                if uploaded_spec: ref_files.extend(uploaded_spec)
                if uploaded_recipe: ref_files.extend(uploaded_recipe)
                
                json_result, chunk_list = analyze_design_with_ai(main_img_objs, ref_files, uploaded_master_fact, legal_knowledge_base, final_db_context_text)
                report_data = json.loads(json_result)
                
                st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
                
                critical_cnt = sum(1 for r in report_data if r.get("risk_level") == "치명적 위반")
                warning_cnt = sum(1 for r in report_data if r.get("risk_level") == "수정 권고")
                pass_cnt = sum(1 for r in report_data if r.get("risk_level") == "정상")
                
                stat_c1, stat_c2, stat_c3 = st.columns(3)
                with stat_c1: st.markdown(f'<div class="metric-box">🚨 치명적 위반 <br><span class="metric-num" style="color:#dc3545;">{critical_cnt}건</span></div>', unsafe_allow_html=True)
                with stat_c2: st.markdown(f'<div class="metric-box">⚠️ 수정 권고 <br><span class="metric-num" style="color:#f39c12;">{warning_cnt}건</span></div>', unsafe_allow_html=True)
                with stat_c3: st.markdown(f'<div class="metric-box">✅ 정상 구간 <br><span class="metric-num" style="color:#2ecc71;">{pass_cnt}건</span></div>', unsafe_allow_html=True)
                
                st.write("")
                
                for idx, chunk_img in enumerate(chunk_list):
                    st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                    row_col1, row_col2 = st.columns([1, 1])
                    
                    with row_col1:
                        st.image(chunk_img, use_container_width=True)
                        
                    with row_col2:
                        issues = [r for r in report_data if r.get("image_index") == idx]
                        if not issues:
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>해당 구간 범용 법적 테두리 및 팩트 확인 완료.</div>', unsafe_allow_html=True)
                        else:
                            for issue in issues:
                                risk = issue.get("risk_level", "정상")
                                css_class = "risk-critical" if risk == "치명적 위반" else "risk-warning" if risk == "수정 권고" else "risk-pass"
                                icon = "❌" if risk == "치명적 위반" else "⚠️" if risk == "수정 권고" else "✅"
                                
                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                - **상세페이지 상태:** {issue.get("marketing_text", "-")}
                                - **QC 대조 기준:** {issue.get("fact_or_legal_ground", "-")}
                                - **분석 및 조치:** {issue.get("discrepancy_analysis", "")}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.write("") 
                    st.markdown("---")

            except json.JSONDecodeError: st.error("AI 응답을 구조화하는 데 실패했습니다. 잠시 후 다시 시도해 주십시오.")
            except Exception as e: st.error(f"서버 과부하 오류가 발생했습니다. 잠시 대기 후 다시 시도해 주십시오. (에러: {e})")
