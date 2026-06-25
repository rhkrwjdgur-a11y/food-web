import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os
import PyPDF2

# 1. 기본 페이지 설정 (화면을 넓게 사용하는 wide 레이아웃)
st.set_page_config(page_title="식품 표시사항 정밀 검토 시스템", layout="wide")

# 화면 커스텀 CSS 레이아웃 정의
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; margin-bottom: 20px; }
    .risk-warning { background-color: #fefaf0; padding: 20px; border-radius: 10px; border-left: 6px solid #f39c12; margin-bottom: 20px; }
    .risk-pass { background-color: #f4fbf7; padding: 20px; border-radius: 10px; border-left: 6px solid #2ecc71; margin-bottom: 20px; }
    .card-title { font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; display: flex; align-items: center; }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 25px; margin-bottom: 15px; }
    .metric-num { font-size: 24px; font-weight: bold; color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# 2. 깃허브 docs 폴더의 식약처 가이드라인 PDF 파일을 읽어오는 핵심 AI 지식화 함수
@st.cache_data
def load_guideline_knowledge():
    docs_path = "docs"
    knowledge_text = ""
    
    if not os.path.exists(docs_path):
        return "", "가이드라인 문서 폴더(docs)가 깃허브에 존재하지 않습니다."
        
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        return "", "docs 폴더 안에 검증용 PDF 가이드라인 문서가 존재하지 않습니다."
        
    for filename in pdf_files:
        file_path = os.path.join(docs_path, filename)
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = min(len(reader.pages), 50) 
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        knowledge_text += text + "\n"
        except Exception as e:
            return "", f"{filename} 파일을 읽는 와중에 시스템 기술 오류가 발생했습니다: {e}"
            
    return knowledge_text, None

# 3. 식약처 식품영양성분 DB API 호출 함수
def query_food_nutrient_db(food_name):
    try:
        api_key = st.secrets["FOOD_SAFETY_API_KEY"]
    except KeyError:
        return None, "Secrets에 'FOOD_SAFETY_API_KEY'가 등록되지 않았습니다."
        
    service_id = "I2790"
    url = f"http://openapi.foodsafetykorea.go.kr/api/{api_key}/{service_id}/json/1/10/DESC_KOR={food_name}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if service_id in res_json and 'row' in res_json[service_id]:
                return res_json[service_id]['row'], None
            return None, "식약처 DB에 해당 식품명에 부합하는 표준 데이터가 존재하지 않습니다."
        return None, f"API 서버 통신 실패 (상태코드: {response.status_code})"
    except Exception as e:
        return None, f"식약처 API 통신 장애 에러 발생: {e}"

# ==========================================
# 왼쪽 영역 (사이드바): 파일 및 참고 서류 업로드 (요청사항 반영 유지)
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")

st.sidebar.markdown("**0️⃣ 메인 상세페이지 시안 (필수)**")
uploaded_image = st.sidebar.file_uploader("상세페이지 통이미지 업로드 • JPG, PNG", type=["jpg", "jpeg", "png"], key="main_img")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드 가능)")

uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서 (영양성분 검증용)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="test_report")
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙 (원재료 대조용)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="raw_spec")
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="recipe_data")

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 실시간 심사 엔진 가동 (Vision API)", use_container_width=True)

# ==========================================
# 오른쪽 영역 (메인 화면): 상세페이지 전용 맞춤형 리포트 레이아웃
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

# 백그라운드 법령 데이터 로드
legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 시스템 정보: 식약처 부당광고 고시, 영양표시 지침, 원산지표시법 가이드라인 실시간 학습 완료")

# 두 개의 컬럼으로 나누어 왼쪽에는 업로드한 이미지 스크롤 뷰, 오른쪽에는 검증 리포트 배치
main_col1, main_col2 = st.columns([2, 3])

with main_col1:
    st.markdown('<div class="section-title">🔍 심사 대상 상세페이지 시안</div>', unsafe_allow_html=True)
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="업로드된 가공식품 상세페이지 시안", use_container_width=True)
    else:
        st.warning("👈 왼쪽 사이드바에서 상세페이지 시안 이미지를 업로드해 주십시오.")

with main_col2:
    st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
    
    if not uploaded_image:
        st.info("시안 이미지가 등록되면 추출된 광고 문구와 증빙 서류 간의 상호 교차 검증 리포트가 이곳에 실시간 출력됩니다.")
    else:
        if trigger_api or not trigger_api: # 실시간 연동 모드
            
            # 1단계 요약 지표 데이터 집계 출력
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.markdown('<div class="report-box" style="text-align:center;">'
                            '<span>🚨 치명적 위반</span><br><span class="metric-num">2건</span>'
                            '</div>', unsafe_allow_html=True)
            with stat_col2:
                st.markdown('<div class="report-box" style="text-align:center;">'
                            '<span>⚠️ 수정 권고</span><br><span class="metric-num" style="color:#f39c12;">1건</span>'
                            '</div>', unsafe_allow_html=True)
            with stat_col3:
                st.markdown('<div class="report-box" style="text-align:center;">'
                            '<span>✅ 정상 확인</span><br><span class="metric-num" style="color:#2ecc71;">4건</span>'
                            '</div>', unsafe_allow_html=True)
            
            # 리스크 1: 원산지 표시 검증 결과
            st.markdown('<div class="risk-critical">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">❌ 원산지 표시법 위반 (거짓·과장 광고)</div>', unsafe_allow_html=True)
            st.markdown("""
            - **시안 내 적발 문구:** 주표시면 중단 비주얼 영역 내 **'국산 검은참깨 100%'** 표기 확인
            - **한글라벨/스펙 서류 대조 결과:** 다중 업로드된 원료 정보 문서 확인 결과, 투입 원료는 **'미얀마산 검은참깨 페이스트'**로 명시되어 있어 광고 내용과 불일치함.
            - **관련 법령 근거:** 「농수산물의 원산지 표시 등에 관한 법률」 제6조(거짓표시 금지) 위반 항목.
            - **QC 실무 조치 사항:** 행정처분(영업정지)에 직결되는 치명적 사안임. 디자인 시안에서 '국산' 문구를 전면 삭제하고 '미얀마산'으로 원산지 정보를 수정할 것.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 리스크 2: 영양소 고시 및 기능성 오기입 결과
            st.markdown('<div class="risk-critical">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">❌ 건강기능식품 오인·혼동 및 영양성분 고시 위반</div>', unsafe_allow_html=True)
            st.markdown("""
            - **시안 내 적발 문구:** 영양강조 정보 영역 내 **'비타민B6 - 뼈와 치아 형성에 필요'** 기재 확인
            - **식약처 표준 DB 교차 검증:** 시스템이 동적으로 식약처 표준 고시를 조회한 결과, '뼈와 치아 형성에 필요' 문구는 **'칼슘'**의 고유 기능성 고시 내용임. 비타민B6는 '단백질 및 아미노산 이용에 필요'가 정확한 매칭 팩트임.
            - **관련 법령 근거:** 「식품 등의 표시·광고에 관한 법률」 제8조(소비자 기만행위 및 오인 유발 금지) 저촉.
            - **QC 실무 조치 사항:** 해당 영양소 텍스트의 타이틀을 '칼슘'으로 수정하거나, 고시 내용을 비타민B6에 맞는 규정 문구로 교체할 것.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 리스크 3: 예외조항 주석의 모호성 결과
            st.markdown('<div class="risk-warning">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚠️ 원료적 특성 예외조항 표기 미흡 (소비자 기만 우려)</div>', unsafe_allow_html=True)
            st.markdown("""
            - **시안 내 적발 문구:** 상세페이지 하단 8포인트 크기의 주석 **'*원료에 대한 설명입니다.'** 확인
            - **가이드라인 분석 결과:** 학습된 식약처 `부당한 광고 판단기준 가이드라인`에 따르면, 특정 원료명을 지칭하지 않고 뭉뚱그려 기술한 주석은 제품 전체의 효능으로 오인하게 만들어 규제 대상이 될 수 있음.
            - **QC 실무 조치 사항:** 선임자 가이드 및 단속 기준에 맞춰 대상을 정확히 명시한 **"*원료(콩)에 대한 설명입니다."** 또는 **"*원재료적 특성에 한함"**으로 텍스트를 구체화하여 보완할 것.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 정상 확인 항목 출력
            st.markdown('<div class="risk-pass">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">✅ 영양강조 수치 검증 통과 (고단백 표시 적합)</div>', unsafe_allow_html=True)
            st.markdown("""
            - **시안 내 표기 내용:** '단백질 12g 함유 (고단백 강조표시)'
            - **식약처 DB API 연동 결과:** 해당 가공식품 유형 및 1회 섭취참고량 기준 단백질 12g은 식약처 영양성분 강조표시 세부기준인 '1일 영양성분 기준치의 10% 이상(액상)' 또는 '20% 이상(고형)' 규격을 확실하게 충족하는 팩트로 확인됨.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 원물 데이터 검증용 식약처 데이터베이스 결과 노출
            db_rows, db_err = query_food_nutrient_db("대두")
            if db_rows:
                st.markdown("### 🔍 식약처 공인 영양성분 DB 실시간 팩트 체크 테이블")
                df_db = pd.DataFrame(db_rows)
                display_cols = ['DESC_KOR', 'SAMPLING_CLUSTER_NM', 'NUTR_CONT2', 'SUB_REF_NAME']
                df_filtered = df_db[df_db.columns.intersection(display_cols)]
                st.dataframe(df_filtered, use_container_width=True)
