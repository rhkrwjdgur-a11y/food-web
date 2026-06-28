import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2

# 1. 기본 페이지 설정
st.set_page_config(page_title="식품 표시사항 정밀 검토 시스템", layout="wide")

# CSS 디자인
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .report-box { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 25px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# 2. API 키 설정 (구글 Gemini API 연결)
try:
    genai.configure(api_key=st.secrets["AI_VISION_API_KEY"])
except KeyError:
    st.error("시스템 오류: Secrets에 'AI_VISION_API_KEY'가 설정되지 않았습니다.")

# 3. 법령 가이드라인 PDF 로드 함수
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
                num_pages = min(len(reader.pages), 20) # 속도 최적화를 위해 핵심 페이지만 추출
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        knowledge_text += text + "\n"
        except Exception as e:
            return "", f"오류 발생: {e}"
            
    return knowledge_text, None

# 4. 실시간 AI 비전 분석 로직 (Gemini 1.5 Pro 모델 호출)
def analyze_design_with_ai(image_obj, legal_text):
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    당신은 엄격한 품질관리(QC) 및 표시광고 검토 전문가입니다.
    업로드된 식품 상세페이지 시안 이미지를 정밀 스캔하고, 아래 제공된 식약처 가이드라인과 교차 대조하십시오.
    
    [식약처 가이드라인 지식 베이스]
    {legal_text}
    
    다음 항목들을 집중적으로 검토하여 리포트를 작성하십시오:
    1. 영양강조표시 위반 (예: 근거 없는 고단백, 저당 표기)
    2. 과대광고 및 건강기능식품 오인 혼동 문구 (질병 치료 암시 등)
    3. 주석 및 예외조항의 모호성 (소비자 기만 우려)
    
    발견된 리스크를 '치명적 위반' 또는 '수정 권고'로 분류하고, 실무자가 수정해야 할 팩트를 명확하게 제시하십시오.
    """
    
    response = model.generate_content([image_obj, prompt])
    return response.text

# ==========================================
# 왼쪽 사이드바: 심사 대상 파일 등록
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_image = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (필수)", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류")
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서", type=["pdf", "jpg", "png"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["pdf", "jpg", "png"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 실시간 심사 엔진 가동 (Vision API)", use_container_width=True)

# ==========================================
# 오른쪽 메인 화면: AI 분석 리포트
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 식약처 부당광고 고시 및 영양표시 지침 실시간 학습 완료")

main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.markdown('<div class="section-title">🔍 심사 대상 상세페이지 시안</div>', unsafe_allow_html=True)
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="업로드된 시안", use_container_width=True)
    else:
        st.warning("👈 왼쪽에서 상세페이지 시안 이미지를 업로드해 주십시오.")

with main_col2:
    st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
    
    if trigger_api:
        if not uploaded_image:
            st.error("이미지를 먼저 업로드해야 분석이 가능합니다.")
        else:
            with st.spinner("구글 Vision API 가동 중: 이미지 내 텍스트 추출 및 법령 대조를 진행하고 있습니다 (약 10~20초 소요)..."):
                try:
                    # 팩트: 여기서 업로드된 이미지를 직접 AI 모델로 전송하여 진짜 분석을 수행합니다.
                    ai_report = analyze_design_with_ai(img, legal_knowledge_base)
                    
                    st.success("✅ 시안 스캔 및 법령 가이드라인 대조 완료")
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.write(ai_report)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"AI 분석 중 오류가 발생했습니다. API 키 설정이나 이미지 용량을 확인하십시오. 상세 에러: {e}")
    else:
        st.info("좌측 하단의 '실시간 심사 엔진 가동' 버튼을 누르면 AI 분석이 시작됩니다.")
