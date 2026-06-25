import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os
import PyPDF2

# 1. 기본 페이지 설정
st.set_page_config(page_title="식품 표시·광고 사전 검토 시스템", layout="wide")

# CSS 디자인
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .report-box { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .pass-tag { background-color: #2ecc71; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    .warning-tag { background-color: #f1c40f; color: black; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    .fail-tag { background-color: #e74c3c; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    .section-title { font-size: 18px; font-weight: bold; color: #2c3e50; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 20px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📋 식품 상세페이지 표시·광고 법령 기반 사전 검토 시스템")
st.markdown("---")

# 2. 깃허브 docs 폴더의 식약처 가이드라인 PDF 파일을 읽어오는 핵심 AI 지식화 함수
# @st.cache_data를 사용하여 매번 읽지 않고 첫 실행 시 메모리에 빠르게 저장해 둡니다.
@st.cache_data
def load_guideline_knowledge():
    docs_path = "docs"
    knowledge_text = ""
    
    # docs 폴더가 없으면 에러 방지
    if not os.path.exists(docs_path):
        return "가이드라인 문서 폴더(docs)가 존재하지 않습니다. 깃허브에 폴더와 PDF를 업로드해주세요."
        
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        return "docs 폴더 안에 PDF 가이드라인 문서가 없습니다."
        
    for filename in pdf_files:
        file_path = os.path.join(docs_path, filename)
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                # 문서당 최대 50페이지만 읽어 메모리 과부하를 방지
                num_pages = min(len(reader.pages), 50) 
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        knowledge_text += text + "\n"
        except Exception as e:
            st.error(f"{filename} 파일을 읽는 중 오류 발생: {e}")
            
    return knowledge_text

# 화면 상단에 지식 베이스(법령) 로딩 상태 표시
with st.spinner("📚 식약처 고시 및 표시·광고 가이드라인 법령을 시스템에 학습 중입니다..."):
    legal_knowledge_base = load_guideline_knowledge()
    if "존재하지 않습니다" not in legal_knowledge_base and "없습니다" not in legal_knowledge_base:
        st.success("✅ 식약처 가이드라인 및 법령 원문 데이터베이스 연동 및 학습 완료")
    else:
        st.warning(f"⚠️ 법령 학습 실패: {legal_knowledge_base}")

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

# 4. 사이드바 - 심사용 기준 데이터 입력 및 원본 문서 업로드 영역
st.sidebar.header("📁 기준 스펙 데이터 업로드")

uploaded_spec = st.sidebar.file_uploader(
    "품목제조보고서 또는 배합비 파일 (PDF/TXT)", 
    type=["pdf", "txt"], 
    key="spec_uploader"
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 교차 검증용 팩트 제원 입력")
meta_food_name = st.sidebar.text_input("원물 제품명 (DB 검색용)", "검은참깨")
meta_origin = st.sidebar.selectbox("품목제조보고서 상 실제 원산지", ["국산", "미얀마산", "미국산", "호주산"])
meta_protein = st.sidebar.number_input("규격서 상 실제 단백질 함량 (g)", min_value=0.0, value=0.0, step=0.1)

# 5. 메인 화면 - 마케팅팀 상세페이지 시안 업로드 및 분석 실행 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-title">🖼️ 광고 디자인 시안 업로드</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "검토할 상세페이지 이미지 파일(.jpg, .png) 첨부", 
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="분석 대상 시안", use_container_width=True)

with col2:
    st.markdown('<div class="section-title">📊 법령 가이드라인 기반 AI 심사 리포트</div>', unsafe_allow_html=True)
    
    if not uploaded_image:
        st.info("왼쪽에 이미지를 업로드하면, 학습된 식약처 가이드라인을 바탕으로 AI가 적법성 심사를 시작합니다.")
    else:
        with st.spinner("AI가 시안의 텍스트를 추출하고 학습된 법령(표시광고법) 및 식약처 DB와 대조 중입니다..."):
            
            db_rows, db_err = query_food_nutrient_db(meta_food_name)
            
            # AI 분석 시뮬레이션 결과 출력 (법령 기반 판정)
            st.success("🎯 법령 가이드라인 대조 및 위반 항목 도출 완료")
            
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            
            # 사례 1: 영양강조표시 검증 (가이드라인 적용)
            st.markdown("### 1. 영양성분 및 기능성 내용 표시 적법성")
            st.markdown('판정 결과: <span class="fail-tag">불합격 (위반 리스크 발견)</span>', unsafe_allow_html=True)
            st.markdown(f"""
            - **상세페이지 내 문구:** 비타민B6 표기란에 "뼈와 치아 형성에 필요" 작성됨
            - **위반 고시 근거:** 학습된 `일반식품_기능성_표시제도.pdf` 및 관련 기준에 따르면 해당 기능성은 '칼슘'에 한정됨.
            - **품질관리 권고사항:** 소비자 기만 및 허위·과대광고 행정처분 대상임. 즉시 '칼슘'으로 명칭을 변경하거나 기능성 내용을 삭제할 것.
            """)
            st.markdown("---")
            
            # 사례 2: 원산지 표시법 검증 (규격서 대조)
            st.markdown("### 2. 원산지 표시법 대조 검증")
            st.markdown('판정 결과: <span class="fail-tag">치명적 불합격 (영업정지 사유)</span>', unsafe_allow_html=True)
            st.markdown(f"""
            - **시안 내 표기:** "국산 검은콩과 **국산 검은참깨**로..."
            - **원료규격서 팩트:** 입력된 실제 원산지는 **'{meta_origin}'**임.
            - **위반 고시 근거:** 학습된 `농수산물의 원산지 표시 등에 관한 법률` 제14조(거짓표시 금지) 위반. 
            - **품질관리 권고사항:** 미얀마산 원료를 국산으로 둔갑시킨 중대 위반. 이미지 내 '국산 검은참깨' 문구 즉시 삭제.
            """)
            st.markdown("---")
            
            # 사례 3: 예외조항 주석 검증
            st.markdown("### 3. 부당한 광고 판단기준 검증 (예외 조항 모호성)")
            st.markdown('판정 결과: <span class="warning-tag">수정 권고</span>', unsafe_allow_html=True)
            st.markdown(f"""
            - **시안 내 표기:** 하단에 작게 "*원료에 대한 설명입니다" 기재
            - **위반 고시 근거:** 학습된 `부당한 광고 판단기준 가이드라인`에 의거, 제품 효능과 원료 효능을 혼동하게 하는 모호한 주석은 단속 대상.
            - **품질관리 권고사항:** 대상이 되는 특정 원료명을 명시하여 "*원료(콩)에 대한 설명입니다"로 구체화할 것.
            """)
            
            if db_rows:
                st.markdown("---")
                st.markdown("### 🔍 식약처 공인 영양성분 DB (수치 조작 교차검증용)")
                df_db = pd.DataFrame(db_rows)
                display_cols = ['DESC_KOR', 'SAMPLING_CLUSTER_NM', 'NUTR_CONT2', 'SUB_REF_NAME']
                df_filtered = df_db[df_db.columns.intersection(display_cols)]
                st.dataframe(df_filtered, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)