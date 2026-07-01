import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2
import json
import time
import requests
import urllib.parse

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

# 3-2. 식약처 영양성분 DB 호출 함수 (인코딩 및 투망식 50개 검색 적용)
def query_food_nutrient_db(food_name):
    if not food_name: return None
    service_id = "I2790"
    # 한글 검색어 URL 인코딩 처리
    encoded_food = urllib.parse.quote(food_name.strip())
    # 검색 범위를 상위 5개에서 50개로 늘려 세부 부위(구운것, 말린것 등)가 모두 포함되도록 수정
    url = f"http://openapi.foodsafetykorea.go.kr/api/{FOOD_API_KEY}/{service_id}/json/1/50/DESC_KOR={encoded_food}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if service_id in res_json and 'row' in res_json[service_id]:
                return res_json[service_id]['row']
    except Exception: pass
    return None

# 4-1. Auto Pre-Scan: 이미지에서 비교 대상 식품 키워드 자동 추출
def auto_extract_db_keywords(main_images):
    model = genai.GenerativeModel('gemini-2.5-flash')
    payload = []
    for img in main_images:
        w, h = img.size
        if w > 1000: img = img.resize((1000, int(h * (1000.0/w))), Image.LANCZOS)
        payload.append(img)
    prompt = """
    당신은 식품 상세페이지에서 외부 영양성분 DB 검색이 필요한 '핵심 원물 명칭'을 자동 추출하는 AI입니다.
    이미지들을 훑어보고, 타 식품과 영양성분을 비교하는 인포그래픽이 있는지 찾으십시오.
    발견되었다면, 검색 API의 호환성을 위해 수식어(구운것, 말린것 등)를 제외한 **가장 기본이 되는 핵심 명사(예: 쇠고기, 닭고기, 대두, 우유)**만을 쉼표(,)로 구분하여 출력하십시오. (세부 분류는 메인 AI가 나중에 대조할 것입니다.)
    없다면 오직 'NONE'이라고만 출력하십시오.
    """
    payload.append(prompt)
    try:
        response = model.generate_content(payload)
        res_text = response.text.strip()
        if res_text == "NONE" or not res_text: return []
        return [k.strip() for k in res_text.split(",")]
    except Exception: return []

# 4-2. 실시간 AI 비전 분석 로직 (세부 부위 매칭 및 적합 판정 명시 룰 탑재)
def analyze_design_with_ai(main_images, ref_files, master_fact_files, legal_text, db_context_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_payload = []
    chunk_list = []
    split_height = 3000
    for img_obj in main_images:
        w, h = img_obj.size
        if w > 2000: img_obj = img_obj.resize((2000, int(h * (2000.0/w))), Image.LANCZOS); w, h = img_obj.size
        for i in range(0, h, split_height):
            chunk = img_obj.crop((0, i, w, min(i + split_height, h)))
            chunk_list.append(chunk)
            content_payload.append(chunk)
            
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
    당신은 엄격한 품질관리(QC) 전문가입니다. 각 시안 조각(인덱스)마다 아래 4가지 항목을 무조건 검토하십시오.
    
    [식약처 법령 지식 베이스]
    {legal_text}
    
    [자동 추출된 국가 공인 영양성분 DB 데이터 (최대 50건 목록)]
    {db_context_text if db_context_text else "검색된 외부 DB 데이터 없음."}
    
    [필수 강제 체크리스트 - 스킵 절대 금지]
    🔥 1. DB 비교 수치 초정밀 검증 (세부 부위 및 조리 상태 완벽 대조):
       - 시안의 비교 그래프에 적힌 작은 글씨(예: '한우, 등심 구운것', '노란콩 말린것', '구운것')를 한 글자도 빠짐없이 읽어내십시오.
       - 제공된 [식약처 DB 데이터] 중에서, 시안에 적힌 부위 및 조리 상태와 완벽하게 일치하는 항목(DESC_KOR)을 찾아내어 단백질 등의 수치를 대조하십시오.
       - 부위/조리법 조건이 맞고 수치까지 일치한다면 "risk_level": "정상"으로 세팅하고 "세부 항목(예: 쇠고기 한우 등심 구운것 18.9g) DB 수치 정확히 일치하여 사용 적합합니다."라고 명시하십시오. 틀리면 "치명적 위반"으로 적발하십시오.

    🔥 2. 칼로리 스팸 방지 및 내부 모순:
       - 상/하단 칼로리가 불일치한다면, 그 에러는 오직 '하단 영양정보표'가 보이는 조각에서만 딱 1번 적발하십시오.

    🔥 3. 기만행위 (시점 조작, 100% 꼼수, 제조공정):
       - 미래 데이터를 바탕으로 1위를 주장하는지, '100% 약콩'이라 해놓고 원재료엔 대두가 섞여있는지 확인하십시오.
       
    🔥 4. 법적 용어 (ZERO vs 무첨가) 및 연출 사진:
       - 콩 등 천연 당류가 존재함에도 '설탕 ZERO'라고 기재했다면 '설탕 무첨가'로 수정 권고하십시오.
       - 연출 사진 주변에 면책 문구 유무를 적발하십시오.
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오.
    [
      {{
        "image_index": 구간 인덱스 번호 (0부터 시작),
        "risk_level": "치명적 위반" 또는 "수정 권고" 또는 "정상",
        "title": "검토 항목 요약",
        "marketing_text": "상세페이지 추출 원문 (예: 쇠고기(한우, 등심 구운것))",
        "fact_or_legal_ground": "팩시안 원문, 외부 DB 매칭 항목, 또는 법적 가이드라인",
        "discrepancy_analysis": "DB 일치 시 세부명칭 포함 적합 판정 문구 기재, 불일치/위반 시 조치 사항 기재"
      }}
    ]
    * 해당 구간에 보고할 위반사항이나 DB 적합 판정이 전혀 없다면 risk_level "정상" 객체(내용: 특이사항 없음)를 반환하십시오.
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
    st.warning("👈 왼쪽 메뉴에서 메인 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
        
    if not trigger_api:
        st.info("좌측 하단의 심사 가동 버튼을 누르면 AI 분석이 시작됩니다.")
        for img in main_img_objs: st.image(img, use_container_width=True)
    else:
        db_context_text = ""
        
        with st.spinner("🔍 1단계: 시안 내 식약처 DB 비교광고 키워드를 자동 탐지하고 있습니다..."):
            auto_keywords = auto_extract_db_keywords(main_img_objs)
            if auto_keywords:
                st.sidebar.success(f"🤖 AI 자동 탐지 핵심 키워드: {', '.join(auto_keywords)}")
                for kw in auto_keywords:
                    db_data = query_food_nutrient_db(kw)
                    if db_data:
                        # 데이터 전체를 넘기되 너무 길면 자름 (토큰 한도 방지)
                        db_context_text += f"\n[검색어 '{kw}' 식약처 공인 데이터 (최대 50건)]\n" + json.dumps(db_data[:50], ensure_ascii=False) + "\n"
                        st.sidebar.info(f"✅ 식약처 DB '{kw}' 연동 완료 (상세 분류 포함 데이터 획득)")
                    else:
                        st.sidebar.error(f"❌ DB에서 '{kw}' 데이터를 찾을 수 없습니다.")
            else:
                st.sidebar.info("🔍 탐지된 비교광고 외부 DB 키워드 없음")

        with st.spinner("⚙️ 2단계: 3-Pass 투트랙 정밀 심사 가동 중 (DB 세부 부위 매칭 및 적합 판정 룰 적용)..."):
            try:
                ref_files = []
                if uploaded_test: ref_files.extend(uploaded_test)
                if uploaded_spec: ref_files.extend(uploaded_spec)
                if uploaded_recipe: ref_files.extend(uploaded_recipe)
                
                json_result, chunk_list = analyze_design_with_ai(main_img_objs, ref_files, uploaded_master_fact, legal_knowledge_base, db_context_text)
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
