import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2
import json
import time
import requests

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

# 2. API 키 설정 (Vision API & 식약처 DB API)
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
    
    if not os.path.exists(docs_path):
        return "", "가이드라인 문서 폴더(docs)가 없습니다."
        
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        return "", "docs 폴더 안에 검증용 PDF 문서가 없습니다."
        
    for filename in pdf_files:
        file_path = os.path.join(docs_path, filename)
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = min(len(reader.pages), 20)
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        knowledge_text += text + "\n"
        except Exception as e:
            return "", f"오류 발생: {e}"
            
    return knowledge_text, None

# 3-2. 식약처 영양성분 DB(I2790) 호출 함수 추가
def query_food_nutrient_db(food_name):
    if not food_name:
        return None
    service_id = "I2790"
    url = f"http://openapi.foodsafetykorea.go.kr/api/{FOOD_API_KEY}/{service_id}/json/1/5/DESC_KOR={food_name}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if service_id in res_json and 'row' in res_json[service_id]:
                return res_json[service_id]['row']
        return None
    except Exception:
        return None

# 4. 실시간 AI 비전 분석 로직 (식약처 DB 컨텍스트 통합)
def analyze_design_with_ai(main_images, ref_files, master_fact_files, legal_text, db_context_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_payload = []
    chunk_list = []
    
    split_height = 3000
    for img_obj in main_images:
        width, height = img_obj.size
        if width > 2000:
            ratio = 2000.0 / width
            img_obj = img_obj.resize((2000, int(height * ratio)), Image.LANCZOS)
            width, height = img_obj.size

        for i in range(0, height, split_height):
            box = (0, i, width, min(i + split_height, height))
            chunk = img_obj.crop(box)
            chunk_list.append(chunk)
            content_payload.append(chunk)
            
    if ref_files:
        for ref in ref_files:
            try:
                ref.seek(0)
                ref_img = Image.open(ref)
                content_payload.append(ref_img)
            except:
                pass 

    master_fact_count = 0
    if master_fact_files:
        for fact in master_fact_files:
            try:
                fact.seek(0)
                fact_img = Image.open(fact)
                content_payload.append(fact_img)
                master_fact_count += 1
            except:
                pass
                
    prompt = f"""
    당신은 엄격한 품질관리(QC) 전문가이자 법적 규제 준수(Compliance) 검토자입니다.
    전송된 이미지 구조는 다음과 같습니다:
    1. 첫 {len(chunk_list)}장: 마케팅 부서가 기획한 '상세페이지 시안' 구간 이미지 (인덱스 0부터 시작)
    2. 그 다음 {master_fact_count}장: 품질관리팀이 승인한 '확정 팩시안(패키지 전개도, 한글표시사항)'
    3. 나머지: 기타 증빙 서류
    
    [식약처 법령 및 가이드라인 지식 베이스]
    {legal_text}
    
    [실시간 국가 공인 영양성분 DB 검색 결과]
    {db_context_text if db_context_text else "검색된 외부 DB 데이터 없음."}
    
    [법적 리스크 및 팩트 대조 절대 룰 - 환각(Hallucination) 영구 차단]
    Rule 1 (시각적 팩트 절대주의): 확정 팩시안의 텍스트를 눈으로 확인한 내용만 기재하십시오. 
    Rule 2 (투트랙 검증 엄수): Track 1(마케팅 문구와 팩시안 수치 100% 일치 여부), Track 2(식약처 부당광고 기준 위반 여부)를 모두 스캔하십시오.
    
    🔥 Rule 3 (비교광고 식약처 DB 철저 검증): 마케팅 문구에서 "소고기보다 단백질이 높다", "우유 칼슘의 X배" 등 타 식품과 비교하는 문구가 있다면, 위 [실시간 국가 공인 영양성분 DB 검색 결과]의 100g당 수치와 비교하여 정확한 수학적 팩트 체크를 진행하십시오. 수치가 과장되었거나 거짓이라면 '치명적 위반(부당한 비교 광고)'으로 적발하십시오.
    
    🔥 Rule 4 (마케팅 수식어 법적 현미경 검증): 마케팅 부서가 사용한 '순수', '100%', '무첨가', '프리미엄' 등의 수식어가 확정 팩시안의 원재료명과 충돌하여 기만행위에 해당하는지 검토하십시오.
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오.
    [
      {{
        "image_index": 구간 인덱스 번호 (0부터 시작),
        "risk_level": "치명적 위반", "수정 권고", 또는 "정상",
        "title": "검토 항목 요약",
        "marketing_text": "상세페이지에서 추출한 마케팅 텍스트 원문",
        "fact_or_legal_ground": "확정 팩시안, 식약처 DB 데이터, 또는 법령 지식베이스에서 발췌한 팩트 근거 원문",
        "discrepancy_analysis": "마케팅 문구의 법적 규제 위반 사항 또는 식약처 DB와의 수치 불일치 분석 및 수정 지시 내용"
      }}
    ]
    위반 사항이 없다면 risk_level을 "정상"으로 반환하고 discrepancy_analysis에 '해당 구간 법적 테두리 내 마케팅 문구 확인 완료'라고 기재하십시오.
    """
    
    content_payload.append(prompt)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                content_payload,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            return response.text, chunk_list
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
            raise e

# ==========================================
# 왼쪽 사이드바: 심사 대상 파일 등록
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 식약처 영양성분 DB 실시간 자동 연동")
# 비교 광고 검증을 위한 DB 검색 인풋 추가
db_search_keyword = st.sidebar.text_input("상세페이지 내 비교 대상 식품명 입력", help="예: 소고기, 닭가슴살, 우유 등 (입력 시 API 자동 호출)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="master_fact_uploader")
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서 및 추가 근거 자료", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 3-Pass 투트랙 + 식약처 DB 정밀 심사 가동", use_container_width=True)

# ==========================================
# 최상단: 타이틀 및 지식베이스 로딩 상태
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 식약처 부당광고 고시 및 영양표시 지침 실시간 학습 완료")

# ==========================================
# 메인 화면: 행(Row) 단위 이미지-리포트 1:1 매칭 출력
# ==========================================
if not uploaded_main_images:
    st.warning("👈 왼쪽 메뉴에서 메인 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = []
    for file in uploaded_main_images:
        main_img_objs.append(Image.open(file))
        
    if not trigger_api:
        st.info("좌측 하단의 심사 가동 버튼을 누르면 AI 분석이 시작됩니다.")
        for img in main_img_objs:
            st.image(img, use_container_width=True)
    else:
        with st.spinner("구글 Vision API 가동 중: 마케팅 문구를 법적 테두리 및 식약처 DB 표준 데이터와 정밀 대조하고 있습니다..."):
            try:
                # 식약처 API 동적 호출
                db_context_text = ""
                if db_search_keyword:
                    db_data = query_food_nutrient_db(db_search_keyword)
                    if db_data:
                        db_context_text = f"검색어 '{db_search_keyword}'에 대한 식약처 데이터:\n" + json.dumps(db_data, ensure_ascii=False)
                        st.sidebar.success(f"✅ 식약처 DB '{db_search_keyword}' 데이터 연동 완료")
                    else:
                        st.sidebar.error(f"❌ 식약처 DB에서 '{db_search_keyword}' 데이터를 찾을 수 없습니다.")

                ref_files = []
                if uploaded_test: ref_files.extend(uploaded_test)
                if uploaded_spec: ref_files.extend(uploaded_spec)
                if uploaded_recipe: ref_files.extend(uploaded_recipe)
                
                # DB 데이터를 포함하여 AI 호출
                json_result, chunk_list = analyze_design_with_ai(main_img_objs, ref_files, uploaded_master_fact, legal_knowledge_base, db_context_text)
                report_data = json.loads(json_result)
                
                st.markdown('<div class="section-title">📊 광고 적정성 3-Pass 투트랙 진단 결과</div>', unsafe_allow_html=True)
                
                critical_cnt = sum(1 for r in report_data if r.get("risk_level") == "치명적 위반")
                warning_cnt = sum(1 for r in report_data if r.get("risk_level") == "수정 권고")
                pass_cnt = sum(1 for r in report_data if r.get("risk_level") == "정상")
                
                stat_c1, stat_c2, stat_c3 = st.columns(3)
                with stat_c1:
                    st.markdown(f'<div class="metric-box">🚨 치명적 위반 <br><span class="metric-num" style="color:#dc3545;">{critical_cnt}건</span></div>', unsafe_allow_html=True)
                with stat_c2:
                    st.markdown(f'<div class="metric-box">⚠️ 수정 권고 <br><span class="metric-num" style="color:#f39c12;">{warning_cnt}건</span></div>', unsafe_allow_html=True)
                with stat_c3:
                    st.markdown(f'<div class="metric-box">✅ 정상 구간 <br><span class="metric-num" style="color:#2ecc71;">{pass_cnt}건</span></div>', unsafe_allow_html=True)
                
                st.write("")
                
                for idx, chunk_img in enumerate(chunk_list):
                    st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                    
                    row_col1, row_col2 = st.columns([1, 1])
                    
                    with row_col1:
                        st.image(chunk_img, use_container_width=True)
                        
                    with row_col2:
                        issues = [r for r in report_data if r.get("image_index") == idx]
                        
                        if not issues:
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>해당 구간 법적 테두리 내 마케팅 문구 확인 완료.</div>', unsafe_allow_html=True)
                        else:
                            for issue in issues:
                                risk = issue.get("risk_level", "정상")
                                
                                if risk == "치명적 위반":
                                    css_class, icon = "risk-critical", "❌"
                                elif risk == "수정 권고":
                                    css_class, icon = "risk-warning", "⚠️"
                                else:
                                    css_class, icon = "risk-pass", "✅"
                                
                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                - **상세페이지 원문:** {issue.get("marketing_text", "-")}
                                - **팩트 또는 법령 근거:** {issue.get("fact_or_legal_ground", "-")}
                                - **분석 및 조치:** {issue.get("discrepancy_analysis", "")}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.write("") 
                                
                    st.markdown("---")

            except json.JSONDecodeError:
                st.error("AI 응답을 구조화하는 데 실패했습니다. 잠시 후 다시 시도해 주십시오.")
            except Exception as e:
                st.error(f"AI 분석 중 서버 과부하 오류가 발생했습니다. 잠시 대기 후 다시 시도해 주십시오. (에러: {e})")
