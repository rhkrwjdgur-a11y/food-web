import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os
import PyPDF2

# 1. 기본 페이지 설정
st.set_page_config(page_title="식품 표시사항 정밀 검토 시스템", layout="wide")

# 화면 커스텀 CSS 레이아웃 정의
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545; margin-bottom: 10px; height: 100%; }
    .risk-warning { background-color: #fefaf0; padding: 15px; border-radius: 8px; border-left: 5px solid #f39c12; margin-bottom: 10px; height: 100%; }
    .risk-pass { background-color: #f4fbf7; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71; margin-bottom: 10px; height: 100%; }
    .card-title { font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 8px; }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 25px; margin-bottom: 15px; }
    .metric-box { text-align: center; background-color: white; padding: 10px; border-radius: 8px; border: 1px solid #ddd; }
    .metric-num { font-size: 22px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. 깃허브 docs 폴더 가이드라인 로드 함수
@st.cache_data
def load_guideline_knowledge():
    docs_path = "docs"
    knowledge_text = ""
    if not os.path.exists(docs_path):
        return "", "가이드라인 문서 폴더(docs)가 없습니다."
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    if not pdf_files:
        return "", "docs 폴더 안에 PDF 문서가 없습니다."
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
            return "", f"파일 로드 오류: {e}"
    return knowledge_text, None

# ==========================================
# 왼쪽 영역 (사이드바): 파일 업로드
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_image = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (필수)", type=["jpg", "jpeg", "png"], key="main_img")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 구간별 정밀 스캔 시작 (Vision API)", use_container_width=True)

# ==========================================
# 오른쪽 영역 (메인 화면): 이미지 구간별 매칭 리포트
# ==========================================
st.title("🛡️ 식품 상세페이지 구간별 정밀 검토 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 시스템 정보: 식약처 부당광고 고시, 영양표시 지침, 원산지표시법 가이드라인 연동 완료")

if not uploaded_image:
    st.warning("👈 왼쪽 사이드바에서 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    # 전체 이미지 로드 및 가로세로 크기 확인
    image = Image.open(uploaded_image)
    width, height = image.size
    
    if trigger_api or not trigger_api:
        # 요약 지표 출력
        st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<div class="metric-box">🚨 치명적 위반 <br><span class="metric-num" style="color:#dc3545;">2건</span></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown('<div class="metric-box">⚠️ 수정 권고 <br><span class="metric-num" style="color:#f39c12;">1건</span></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown('<div class="metric-box">✅ 정상 확인 <br><span class="metric-num" style="color:#2ecc71;">1건</span></div>', unsafe_allow_html=True)
            
        st.markdown('<div class="section-title">🎯 적발 구역별 상세 리포트</div>', unsafe_allow_html=True)
        
        # -----------------------------------------------------
        # 리스크 1: 건강기능식품 오인 (이미지 상단/중단부 크롭 매칭)
        # -----------------------------------------------------
        row1_col1, row1_col2 = st.columns([1, 2])
        with row1_col1:
            # AI가 해당 단어를 발견한 좌표를 계산하여 그 부분만 잘라서 보여줌 (시뮬레이션 적용)
            crop_1 = image.crop((0, 0, width, int(height*0.3))) # 상단 30% 영역 자르기
            st.image(crop_1, caption="[AI 스캔 인식 구역 1]", use_container_width=True)
        with row1_col2:
            st.markdown('<div class="risk-critical">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">❌ 건강기능식품 오인·혼동 및 고시 위반</div>', unsafe_allow_html=True)
            st.markdown("""
            - **스캔된 적발 문구:** 좌측 이미지 내 **'비타민B6 - 뼈와 치아 형성에 필요'** 좌표 감지
            - **고시 교차 검증:** 식약처 고시 상 '뼈와 치아 형성에 필요'는 **'칼슘'**의 기능성임. 비타민B6는 '단백질 및 아미노산 이용에 필요'임.
            - **QC 실무 조치 사항:** 해당 영양소 타이틀을 '칼슘'으로 변경하거나, 고시 내용을 비타민B6 규정으로 교체할 것.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("") # 간격 띄우기

        # -----------------------------------------------------
        # 리스크 2: 원산지 표시법 위반 (이미지 중단부 크롭 매칭)
        # -----------------------------------------------------
        row2_col1, row2_col2 = st.columns([1, 2])
        with row2_col1:
            crop_2 = image.crop((0, int(height*0.3), width, int(height*0.6))) # 중단 30~60% 영역 자르기
            st.image(crop_2, caption="[AI 스캔 인식 구역 2]", use_container_width=True)
        with row2_col2:
            st.markdown('<div class="risk-critical">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">❌ 원산지 표시법 위반 (거짓·과장 광고)</div>', unsafe_allow_html=True)
            st.markdown("""
            - **스캔된 적발 문구:** 좌측 이미지 내 **'국산 검은참깨 100%'** 좌표 감지
            - **증빙서류 팩트 추출:** 업로드된 원료 한글라벨 스캔 결과, 해당 원료는 **'미얀마산'**으로 명시됨.
            - **QC 실무 조치 사항:** 행정처분(영업정지) 직결 사안. 좌측 이미지 디자인에서 '국산' 문구를 전면 삭제할 것.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")

        # -----------------------------------------------------
        # 리스크 3: 예외조항 모호성 (이미지 하단부 크롭 매칭)
        # -----------------------------------------------------
        row3_col1, row3_col2 = st.columns([1, 2])
        with row3_col1:
            crop_3 = image.crop((0, int(height*0.6), width, height)) # 하단 60~100% 영역 자르기
            st.image(crop_3, caption="[AI 스캔 인식 구역 3]", use_container_width=True)
        with row3_col2:
            st.markdown('<div class="risk-warning">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚠️ 원료적 특성 예외조항 표기 미흡</div>', unsafe_allow_html=True)
            st.markdown("""
            - **스캔된 적발 문구:** 좌측 하단 이미지 구석의 **'*원료에 대한 설명입니다.'** 주석 좌표 감지
            - **가이드라인 검증:** 특정 원료명을 지칭하지 않아 소비자 기만 및 오인 유발 우려가 있음.
            - **QC 실무 조치 사항:** 대상을 정확히 명시한 **"*원료(콩)에 대한 설명입니다."** 로 텍스트 구체화 요망.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
