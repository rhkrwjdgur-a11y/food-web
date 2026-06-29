import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2
import json

# 1. 기본 페이지 설정
st.set_page_config(page_title="식품 표시사항 정밀 검토 시스템", layout="wide")

# CSS 디자인
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-warning { background-color: #fefaf0; padding: 20px; border-radius: 10px; border-left: 6px solid #f39c12; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .card-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 10px; margin-bottom: 15px; }
    .metric-box { text-align: center; background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
    .metric-num { font-size: 26px; font-weight: bold; margin-top: 5px; display: block; }
    </style>
    """, unsafe_allow_html=True)

# 2. API 키 설정
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
                num_pages = min(len(reader.pages), 20)
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        knowledge_text += text + "\n"
        except Exception as e:
            return "", f"오류 발생: {e}"
            
    return knowledge_text, None

# 4. 실시간 AI 비전 분석 로직 (다중 메인 이미지 지원 및 JSON 출력)
def analyze_design_with_ai(main_images, ref_files, legal_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_payload = []
    
    # 분할된 여러 장의 메인 상세페이지 시안을 모두 페이로드에 추가 (해상도 보존)
    for img_obj in main_images:
        width, height = img_obj.size
        max_height = 10000
        
        if height > 16000:
            for i in range(0, height, max_height):
                box = (0, i, width, min(i + max_height, height))
                chunk = img_obj.crop(box)
                content_payload.append(chunk)
        else:
            content_payload.append(img_obj)
            
    # 보조 증빙 서류 추가
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
    전송된 이미지들은 여러 장으로 나뉘어 업로드된 '메인 상세페이지 시안'들과 팩트 체크를 위한 '증빙 서류'들입니다.
    
    [식약처 가이드라인 지식 베이스]
    {legal_text}
    
    다음 항목들을 집중 검토하십시오:
    1. 원산지/원재료 거짓·과장 (메인 광고와 증빙 서류 팩트 불일치 여부)
    2. 영양강조표시 누락 및 건강기능식품 오인 혼동 문구
    3. 예외조항 주석의 모호성
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오. (이미지 좌표는 필요 없습니다.)
    [
      {{
        "risk_level": "치명적 위반" 또는 "수정 권고",
        "title": "위반 항목 제목",
        "found_text": "발견된 실제 문제 문구",
        "fact_check": "가이드라인 또는 증빙 서류와 대조한 팩트 결과",
        "recommendation": "즉시 수정해야 할 조치 사항"
      }}
    ]
    위반 사항이 없다면 빈 배열 [] 을 반환하십시오.
    """
    
    content_payload.append(prompt)
    
    response = model.generate_content(
        content_payload,
        generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
    )
    
    return response.text

# ==========================================
# 왼쪽 사이드바: 심사 대상 파일 등록
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
# 팩트: 메인 상세페이지 시안도 (상), (하)로 나뉜 파일을 한 번에 올릴 수 있도록 다중 업로드 허용
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 실시간 심사 엔진 가동 (Vision API)", use_container_width=True)

# ==========================================
# 최상단: 타이틀 및 지식베이스 로딩 상태
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 식약처 부당광고 고시 및 영양표시 지침 실시간 학습 완료")

# ==========================================
# 메인 화면 레이아웃 분할 (좌: 원본 미리보기 / 우: AI 리포트)
# ==========================================
if not uploaded_main_images:
    st.warning("👈 왼쪽 메뉴에서 메인 상세페이지 시안 이미지를 하나 이상 업로드해 주십시오.")
else:
    main_col1, main_col2 = st.columns([1, 1])
    
    with main_col1:
        st.markdown('<div class="section-title">🔍 업로드된 전체 상세페이지 시안</div>', unsafe_allow_html=True)
        # 여러 장으로 쪼개진 파일들을 순서대로 모두 크게 띄워줌
        main_img_objs = []
        for file in uploaded_main_images:
            img = Image.open(file)
            main_img_objs.append(img)
            st.image(img, use_container_width=True)

    with main_col2:
        st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
        
        if not trigger_api:
            st.info("좌측 하단의 '실시간 심사 엔진 가동' 버튼을 누르면 AI 분석이 시작됩니다.")
        else:
            with st.spinner("구글 Vision API 가동 중: 정밀 스캔 및 팩트 대조를 진행하고 있습니다 (약 10~20초 소요)..."):
                try:
                    ref_files = []
                    if uploaded_test: ref_files.extend(uploaded_test)
                    if uploaded_spec: ref_files.extend(uploaded_spec)
                    if uploaded_recipe: ref_files.extend(uploaded_recipe)
                    
                    json_result = analyze_design_with_ai(main_img_objs, ref_files, legal_knowledge_base)
                    report_data = json.loads(json_result)
                    
                    if not report_data:
                        st.success("✅ 심사 완료: 식약처 고시 위반 및 증빙 서류 불일치 리스크가 발견되지 않았습니다.")
                    else:
                        critical_cnt = sum(1 for r in report_data if r.get("risk_level") == "치명적 위반")
                        warning_cnt = sum(1 for r in report_data if r.get("risk_level") == "수정 권고")
                        
                        stat_c1, stat_c2, stat_c3 = st.columns(3)
                        with stat_c1:
                            st.markdown(f'<div class="metric-box">🚨 치명적 위반 <br><span class="metric-num" style="color:#dc3545;">{critical_cnt}건</span></div>', unsafe_allow_html=True)
                        with stat_c2:
                            st.markdown(f'<div class="metric-box">⚠️ 수정 권고 <br><span class="metric-num" style="color:#f39c12;">{warning_cnt}건</span></div>', unsafe_allow_html=True)
                        with stat_c3:
                            st.markdown(f'<div class="metric-box">✅ 검토 완료 <br><span class="metric-num" style="color:#2ecc71;">완료</span></div>', unsafe_allow_html=True)
                        
                        st.write("")
                        st.markdown("### 🎯 적발 구역별 상세 리포트")
                        st.write("")
                        
                        # 지저분한 크롭 로직을 제거하고, 텍스트 카드만 시원하게 출력
                        for issue in report_data:
                            risk = issue.get("risk_level", "수정 권고")
                            css_class = "risk-critical" if risk == "치명적 위반" else "risk-warning"
                            icon = "❌" if risk == "치명적 위반" else "⚠️"
                            
                            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                            st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                            st.markdown(f"""
                            - **스캔된 적발 문구:** {issue.get("found_text", "")}
                            - **팩트 교차 검증:** {issue.get("fact_check", "")}
                            - **QC 실무 조치 사항:** {issue.get("recommendation", "")}
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)

                except json.JSONDecodeError:
                    st.error("AI 응답을 구조화하는 데 실패했습니다. 시스템 로그를 확인해 주십시오.")
                except Exception as e:
                    st.error(f"AI 분석 중 오류가 발생했습니다. 상세 에러: {e}")
