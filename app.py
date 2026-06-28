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

# 4. 실시간 AI 비전 분석 로직 (최신 gemini-2.5-flash 모델 적용)
def analyze_design_with_ai(image_obj, ref_files, legal_text):
    # 팩트: 404 에러 원천 차단 및 멀티모달 고속 처리를 위해 gemini-2.5-flash 모델로 변경
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_payload = []
    
    # 메인 상세페이지가 WebP 포맷 한계선인 16000px을 넘을 경우 10000px 단위로 자동 분할(Slicing)
    width, height = image_obj.size
    max_height = 10000
    
    if height > 16000:
        for i in range(0, height, max_height):
            box = (0, i, width, min(i + max_height, height))
            chunk = image_obj.crop(box)
            content_payload.append(chunk)
    else:
        content_payload.append(image_obj)
        
    # 보조 증빙 서류(한글라벨 등)가 추가 업로드된 경우 페이로드에 병합
    if ref_files:
        for ref in ref_files:
            try:
                ref.seek(0)
                ref_img = Image.open(ref)
                content_payload.append(ref_img)
            except:
                pass 
                
    prompt = f"""
    당신은 엄격한 품질관리(QC) 및 표시광고 검토 전문가입니다.
    함께 전송된 이미지들은 '메인 상세페이지 시안'과 원산지/배합비 등을 증명하는 '참고 증빙 서류'들입니다.
    이 이미지들의 텍스트를 정밀 스캔하고, 아래 제공된 식약처 법령 가이드라인과 상호 교차 대조하십시오.
    
    [식약처 가이드라인 지식 베이스]
    {legal_text}
    
    다음 3가지 항목을 집중적으로 검토하여 리포트를 작성하십시오:
    1. 원산지 및 원재료 거짓·과장 (메인 광고 문구와 증빙 서류의 팩트 불일치 여부 정밀 대조)
    2. 영양강조표시 및 건강기능식품 오인 혼동 문구 (질병 치료 암시 등)
    3. 주석 및 예외조항의 모호성 (소비자 기만 우려)
    
    발견된 리스크를 '치명적 위반' 또는 '수정 권고'로 분류하고, 실무자가 즉시 수정할 수 있도록 팩트를 명확하게 제시하십시오.
    """
    
    content_payload.append(prompt)
    response = model.generate_content(content_payload)
    return response.text

# ==========================================
# 왼쪽 사이드바: 심사 대상 파일 등록
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_image = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (필수)", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

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
            st.error("메인 상세페이지 이미지를 먼저 업로드해야 분석이 가능합니다.")
        else:
            with st.spinner("구글 Vision API 가동 중: 통이미지 무손실 분할 및 증빙 서류 교차 검증을 진행하고 있습니다 (약 5~15초 소요)..."):
                try:
                    ref_files = []
                    if uploaded_test: ref_files.extend(uploaded_test)
                    if uploaded_spec: ref_files.extend(uploaded_spec)
                    if uploaded_recipe: ref_files.extend(uploaded_recipe)
                    
                    # AI 분석 수행 (gemini-2.5-flash 엔진 호출)
                    ai_report = analyze_design_with_ai(img, ref_files, legal_knowledge_base)
                    
                    st.success("✅ 시안 스캔 및 법령 가이드라인 대조 완료")
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.write(ai_report)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"AI 분석 중 오류가 발생했습니다. 상세 에러: {e}")
    else:
        st.info("좌측 하단의 '실시간 심사 엔진 가동' 버튼을 누르면 AI 분석이 시작됩니다.")
