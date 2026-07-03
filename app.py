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
import re

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
    table { width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 10px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    th { background-color: #f2f2f2; font-weight: bold; }
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

# 3-2. 식약처 영양성분 DB 호출 함수 (인코딩 완벽 수정 및 JSON 에러 방어)
def query_food_nutrient_db(food_name):
    if not food_name: return None
    
    std_dict = {
        "쇠고기": "소고기",
        "계육": "닭고기",
        "돈육": "돼지고기"
    }
    search_name = std_dict.get(food_name.strip(), food_name.strip())
    
    service_id = "I2790"
    # 식약처 REST Path 방식에 맞춘 단일 URL 인코딩
    encoded_food = urllib.parse.quote(search_name)
    url = f"http://openapi.foodsafetykorea.go.kr/api/{FOOD_API_KEY}/{service_id}/json/1/50/DESC_KOR={encoded_food}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            try:
                res_json = response.json()
            except ValueError:
                st.sidebar.error(f"식약처 서버 응답 오류 (JSON이 아님): {search_name}")
                return None
                
            if 'RESULT' in res_json and res_json['RESULT'].get('CODE') != 'INFO-000':
                st.sidebar.error(f"식약처 API 오류 ({search_name}): {res_json['RESULT'].get('MSG')}")
                return None
            if service_id in res_json and 'row' in res_json[service_id]:
                return res_json[service_id]['row']
    except Exception as e: 
        st.sidebar.error(f"통신 에러 ({search_name}): 서버 연결 실패")
    return None

# 4-1. Auto Pre-Scan
def auto_extract_db_keywords_json(main_images):
    model = genai.GenerativeModel('gemini-2.5-flash')
    payload = []
    for img in main_images:
        w, h = img.size
        if w > 1000: img = img.resize((1000, int(h * (1000.0/w))), Image.LANCZOS)
        payload.append(img)
    prompt = """
    당신은 식품 상세페이지에서 외부 영양성분 DB 대조에 필요한 '검색어'와 '세부 조건'을 추출하는 AI입니다.
    이미지를 훑어보고, 타 식품(쇠고기, 닭고기, 대두 등)과 수치를 비교하는 마케팅 자료가 있다면, 
    API 검색용 '대분류 명사'를 Key로, 시안에 적힌 '세부 수식어(괄호 안 글씨 등)'를 Value로 하는 JSON 객체를 출력하십시오.
    
    [예시] {"쇠고기": "한우, 등심 구운것", "닭고기": "구운것", "대두": "노란콩 말린것"}
    비교 자료가 없다면 오직 "NONE" 이라고만 출력하십시오.
    """
    payload.append(prompt)
    try:
        response = model.generate_content(payload)
        res_text = response.text.strip()
        if res_text == "NONE" or not res_text: return {}
        res_text = re.sub(r'```json\s*', '', res_text)
        res_text = re.sub(r'```\s*', '', res_text)
        return json.loads(res_text)
    except Exception: return {}

# 4-2. 실시간 AI 비전 분석 로직 (비교 대조표 Table 생성 강제 룰 탑재)
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
    당신은 엄격한 품질관리(QC) 전문가입니다. 제공된 시안 조각을 검토하십시오.
    
    [식약처 법령 지식 베이스]
    {legal_text}
    
    [자동 추출된 국가 공인 영양성분 DB 데이터]
    {db_context_text if db_context_text else "경고: 외부 DB 데이터가 존재하지 않습니다."}
    
    [필수 강제 체크리스트 - 스킵 절대 금지]
    
    🔥 1. DB 비교 수치 세부 조건 정밀 검증 및 **[비교 표 생성 강제]**:
       - 시안 내 비교 대상의 세부 수식어(예: 노란콩 말린것)를 읽고, DB 결과 중에서 부합하는 항목의 수치를 대조하십시오.
       - 대조 결과를 반드시 아래 형식의 **마크다운 표(Table)**로 작성하여 `discrepancy_analysis` 항목 첫 줄에 포함시키십시오.
         | 비교 항목 | 시안 표기 수치 | 식약처 DB 실제 수치 | 일치 여부 |
         |---|---|---|---|
         | 쇠고기(등심 구운것) | 18.9g | 18.9g | 일치 (적합) |
       - 만약 DB 데이터가 없다면(경고 문구만 있다면) 절대 지레짐작으로 '적합'이라고 쓰지 말고, 표 안에 "DB 통신 실패로 검증 불가"라고 팩트만 적고 risk_level을 "수정 권고"로 맞추십시오.

    🔥 2. 당류 법적 용어 엄격 구분:
       - 원물 유래 당류가 0.5g 이상 표기되어 있다면 절대 'ZERO'는 쓸 수 없으며 '설탕 무첨가'로 수정 권고하십시오.

    🔥 3. 시간 조작 방어 & 기만 방어:
       - 산정 기간이 현재({current_date_str})를 초과하는 미래인지 대조하십시오.
       - 특정 하위 원료를 '100%' 강조했으나 원재료명에 상위 범주나 타 원료가 섞여있는지 확인하십시오.
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오.
    [
      {{
        "image_index": 구간 인덱스 번호 (0부터 시작),
        "risk_level": "치명적 위반" 또는 "수정 권고" 또는 "정상",
        "title": "검토 항목 요약",
        "marketing_text": "상세페이지 추출 원문",
        "fact_or_legal_ground": "팩시안, 식약처 DB 매칭 항목, 또는 법적 잣대",
        "discrepancy_analysis": "위반 분석 및 조치 사항 (DB 대조 시 반드시 마크다운 표 삽입)"
      }}
    ]
    * 해당 이미지에 위반사항이나 표를 그릴 DB 데이터가 없다면 risk_level "정상" 객체(내용: 특이사항 없음)를 반환하십시오.
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
        
        with st.spinner("🔍 1단계: 시안 내 식약처 DB 타겟(대분류+세부조건)을 자동 추출하고 있습니다..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            if auto_dict:
                for base_food, detail_cond in auto_dict.items():
                    st.sidebar.success(f"🤖 탐지 완료: [{base_food}] ➔ 타겟 조건: {detail_cond}")
                    db_data = query_food_nutrient_db(base_food)
                    if db_data:
                        final_db_context_text += f"\n[검색어 '{base_food}' (시안 내 세부조건: {detail_cond}) 식약처 공인 데이터 최대 50건]\n" + json.dumps(db_data[:50], ensure_ascii=False) + "\n"
                        st.sidebar.info(f"✅ 식약처 DB '{base_food}' 연동 완료")
                    else:
                        st.sidebar.warning(f"⚠️ DB에서 '{base_food}' 데이터를 찾을 수 없거나 통신 실패했습니다.")
            else:
                st.sidebar.info("🔍 탐지된 비교광고 외부 DB 키워드 없음")

        with st.spinner("⚙️ 2단계: 3-Pass 투트랙 정밀 심사 가동 중 (비교 표 Table 강제 생성 중)..."):
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
