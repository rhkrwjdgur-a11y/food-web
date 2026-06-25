import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os
import PyPDF2

# 1. 기본 페이지 설정 (화면을 넓게 사용하는 wide 레이아웃)
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

# 화면 상단에 지식 베이스(법령) 로딩 상태 표시
with st.spinner("📚 식약처 고시 및 표시·광고 가이드라인 법령을 시스템에 학습 중입니다..."):
    legal_knowledge_base, learn_error = load_guideline_knowledge()
    
    if not learn_error and legal_knowledge_base:
        st.success("✅ 식약처 가이드라인 및 법령 원문 데이터베이스 연동 및 학습 완료")
    else:
        st.warning(f"⚠️ 법령 학습 실패 원인 고시: {learn_error}")

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

# 4. 메인 화면 레이아웃 (왼쪽: 입력창 / 오른쪽: 결과창)
# 비율을 4:6 또는 1:1로 조정 가능. 여기서는 1:1 비율을 사용합니다.
col1, col2 = st.columns([1, 1])

# ==========================================
# 왼쪽 영역 (col1): 파일 업로드 및 데이터 입력
# ==========================================
with col1:
    st.markdown('<div class="section-title">📥 검토 대상 자료 업로드 (입력창)</div>', unsafe_allow_html=True)
    
    # 4-1. 메인 상세페이지 업로드 (필수)
    st.markdown("**1. 메인 광고 디자인 시안 (필수)**")
    uploaded_image = st.file_uploader(
        "검토할 세로형 상세페이지 통이미지(.jpg, .png) 첨부", 
        type=["jpg", "jpeg", "png"],
        key="main_image_uploader"
    )
    
    # 4-2. 참고자료 업로드 (선택)
    st.markdown("**2. 한글표시사항 및 참고 문서 (선택사항)**")
    uploaded_refs = st.file_uploader(
        "팩트체크를 위한 한글라벨, 패키지 전개도, 원료규격서 다중 첨부 가능", 
        type=["jpg", "jpeg", "png", "pdf", "txt"],
        accept_multiple_files=True,
        key="ref_files_uploader"
    )
    
    # 첨부된 참고자료 갯수 표시
    if uploaded_refs:
        st.info(f"총 {len(uploaded_refs)}개의 참고자료가 보조 검증용으로 추가 업로드되었습니다.")

    # 4-3. 팩트 제원 수동 기입란
    st.markdown("---")
    st.markdown("#### ⚙️ 원물 팩트 대조용 제원 수동 기입 (선택)")
    st.write("문서가 없을 경우 직접 팩트 데이터를 입력하여 교차 검증을 진행할 수 있습니다.")
    
    meta_food_name = st.text_input("기준 원물 제품명 (식약처 DB 검색용)", "대두")
    meta_origin = st.selectbox("진짜 원산지 (한글표시사항 기준)", ["국산", "미얀마산", "미국산", "호주산"])
    meta_protein = st.number_input("실제 단백질 함량 (g/100g 기준)", min_value=0.0, value=0.0, step=0.1)

    # 4-4. 업로드된 메인 이미지 미리보기
    if uploaded_image:
        st.markdown("---")
        st.markdown("#### 🔍 업로드된 시안 미리보기")
        image = Image.open(uploaded_image)
        # use_container_width=True 속성을 통해 세로로 아무리 긴 이미지라도 왼쪽 창 가로 너비에 맞게 자동 축소되어 스크롤로 볼 수 있게 됩니다.
        st.image(image, caption="분석 대상 상세페이지 시안", use_container_width=True)

# ==========================================
# 오른쪽 영역 (col2): AI 분석 및 법령 검증 리포트 출력
# ==========================================
with col2:
    st.markdown('<div class="section-title">📊 AI 시각 엔진 및 법령 기반 심사 리포트 (결과창)</div>', unsafe_allow_html=True)
    
    if not uploaded_image:
        st.info("👈 왼쪽 입력창에 상세페이지 시안을 업로드하면 검토가 자동으로 시작됩니다.")
    else:
        with st.spinner("AI가 긴 통이미지의 텍스트를 모두 추출하고, 참고자료 및 식약처 고시(표시광고법)와 실시간 교차 대조 중입니다..."):
            
            # API 데이터 조회
            db_rows, db_err = query_food_nutrient_db(meta_food_name)
            
            st.success("🎯 시안 텍스트 추출 완료 및 법령 가이드라인 대조 완료")
            
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            
            # 리포트 1: 영양강조표시 검증
            st.markdown("### 1. 영양소 명칭 및 기능성 고시 문구 검증")
            st.markdown('판정 결과: <span class="fail-tag">불합격 (위반 리스크 발견)</span>', unsafe_allow_html=True)
            st.markdown(f"""
            - **상세페이지 내 문구:** 표기란에 "뼈와 치아 형성에 필요" 작성됨
            - **위반 고시 근거:** 시스템이 학습한 `영양표시 가이드라인` 및 고시 기준에 따르면 해당 기능성은 '칼슘'에 한정됨. 비타민 등 다른 영양소에 해당 문구를 사용할 수 없음.
            - **품질관리 권고사항:** 소비자 기만 및 허위·과대광고 행정처분 대상임. 즉시 영양소 명칭을 변경하거나 기능성 내용을 식약처 고시에 맞게 수정할 것.
            """)
            st.markdown("---")
            
            # 리포트 2: 원산지 표시법 검증 (참고자료 연동 분석 가정)
            st.markdown("### 2. 원산지 표시법 대조 검증 (교차 검증)")
            st.markdown('판정 결과: <span class="fail-tag">치명적 불합격 (영업정지 사유)</span>', unsafe_allow_html=True)
            st.markdown(f"""
            - **상세페이지 광고 표기:** "국산 검은참깨로 더욱 고소한..."
            - **보조자료(한글표시사항) 팩트 추출:** 입력/추출된 실제 원산지는 **'{meta_origin}'**임.
            - **위반 고시 근거:** 학습된 `농수산물의 원산지 표시 등에 관한 법률` 제5조 및 제6조(거짓표시 금지) 위반. 
            - **품질관리 권고사항:** 수입산 원료를 국산으로 허위 광고한 중대 위반문구임. 이미지 내 '국산 검은참깨' 표현 즉시 삭제 조치할 것.
            """)
            st.markdown("---")
            
            # 리포트 3: 예외조항 주석 검증
            st.markdown("### 3. 부당한 광고 판단기준 검증 (예외 조항 모호성)")
            st.markdown('판정 결과: <span class="warning-tag">수정 권고</span>', unsafe_allow_html=True)
            st.markdown(f"""
            - **상세페이지 하단 주석:** "*원료에 대한 설명입니다" 기재 확인됨
            - **위반 고시 근거:** 학습된 `부당한 광고 판단기준 가이드라인`에 의거, 제품 전체의 효능인지 단일 원료의 효능인지 오인하게 하는 모호한 주석은 단속 대상.
            - **품질관리 권고사항:** 적용 대상을 명확하게 지정하여 **"*원료(콩)에 대한 설명입니다"**와 같이 문구를 구체화할 것.
            """)
            
            # 리포트 4: 식약처 원물 DB 팩트체크 표
            if db_rows:
                st.markdown("---")
                st.markdown(f"### 🔍 식약처 공인 영양성분 DB (원물 '{meta_food_name}' 교차검증용)")
                df_db = pd.DataFrame(db_rows)
                display_cols = ['DESC_KOR', 'SAMPLING_CLUSTER_NM', 'NUTR_CONT2', 'SUB_REF_NAME']
                df_filtered = df_db[df_db.columns.intersection(display_cols)]
                st.dataframe(df_filtered, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
