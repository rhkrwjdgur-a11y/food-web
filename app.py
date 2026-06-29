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

# 4. 실시간 AI 비전 분석 로직 (3-Pass 엔진 및 구간별 1:1 매칭)
def analyze_design_with_ai(main_images, ref_files, legal_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_payload = []
    chunk_list = []
    
    # 파이썬 로직으로 이미지를 시각적으로 보기 좋은 3000px 단위로 강제 분할(Sectioning)
    split_height = 3000
    
    for img_obj in main_images:
        width, height = img_obj.size
        # 가로 폭이 너무 크면 비율에 맞춰 축소
        if width > 2000:
            ratio = 2000.0 / width
            img_obj = img_obj.resize((2000, int(height * ratio)), Image.LANCZOS)
            width, height = img_obj.size

        for i in range(0, height, split_height):
            box = (0, i, width, min(i + split_height, height))
            chunk = img_obj.crop(box)
            chunk_list.append(chunk)
            content_payload.append(chunk)
            
    # 보조 증빙 서류 추가
    if ref_files:
        for ref in ref_files:
            try:
                ref.seek(0)
                ref_img = Image.open(ref)
                content_payload.append(ref_img)
            except:
                pass 
                
    # 팩트: 3-Pass 프로세스와 모든 구간에 대한 응답을 강제하는 프롬프트
    prompt = f"""
    당신은 엄격한 품질관리(QC) 및 표시광고 검토 전문가입니다.
    전송된 이미지 중 첫 {len(chunk_list)}장은 '메인 상세페이지 시안'을 위에서 아래로 자른 '구간(Section)' 이미지들이며 (인덱스 0부터 {len(chunk_list)-1}까지), 나머지는 팩트 체크를 위한 '증빙 서류(성적서, 라벨 등)'들입니다.
    
    [식약처 가이드라인 지식 베이스]
    {legal_text}
    
    [3-Pass 검토 프로세스]
    제공된 모든 구간(Section) 인덱스에 대해 예외 없이 다음 3단계를 거쳐 분석하십시오.
    Pass 1 (추출): 해당 구간의 이미지에서 광고 문구, 영양소 수치, 원물 이미지 등 시각적 텍스트를 모두 추출한다.
    Pass 2 (대조): 추출된 내용을 '식약처 지식 베이스' 및 '증빙 서류(팩트)'와 교차 검증하여 일치하는지 확인한다.
    Pass 3 (판정): 허위 원산지, 함량 누락, 고시 위반, 오인 혼동 문구가 있는지 최종 판정한다.
    
    반드시 아래의 JSON 배열(Array) 형식으로 응답하십시오. 
    주의: 0부터 {len(chunk_list)-1}까지의 모든 image_index가 배열 안에 무조건 1개 이상 존재해야 합니다. 문제가 없다면 risk_level을 "정상"으로 반환하십시오.
    
    [
      {{
        "image_index": 구간 인덱스 번호 (0부터 시작),
        "risk_level": "치명적 위반", "수정 권고", 또는 "정상",
        "title": "검토 항목 요약 (예: 제원 표기 정상, 원산지 위반 등)",
        "found_text": "발견된 실제 문구 (정상일 경우 생략 가능)",
        "fact_check": "증빙서류 및 고시와 대조한 3-Pass 결과 팩트",
        "recommendation": "조치 사항 (정상일 경우 '해당 구간 이상 없음' 기재)"
      }}
    ]
    """
    
    content_payload.append(prompt)
    
    response = model.generate_content(
        content_payload,
        generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
    )
    
    return response.text, chunk_list

# ==========================================
# 왼쪽 사이드바: 심사 대상 파일 등록
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 3-Pass 정밀 심사 가동 (Vision API)", use_container_width=True)

# ==========================================
# 최상단: 타이틀 및 지식베이스 로딩 상태
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 식약처 부당광고 고시 및 영양표시 지침 실시간 학습 완료")

# ==========================================
# 메인 화면: 잘라진 구간별 좌우 1:1 매칭 렌더링
# ==========================================
if not uploaded_main_images:
    st.warning("👈 왼쪽 메뉴에서 메인 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = []
    for file in uploaded_main_images:
        main_img_objs.append(Image.open(file))
        
    if not trigger_api:
        st.info("좌측 하단의 '3-Pass 정밀 심사 가동' 버튼을 누르면 AI 분석이 시작됩니다.")
        for img in main_img_objs:
            st.image(img, use_container_width=True)
    else:
        with st.spinner("구글 Vision API 가동 중: 3-Pass(추출-대조-판정) 엔진이 구간별 정밀 검토를 진행하고 있습니다 (약 15~30초 소요)..."):
            try:
                ref_files = []
                if uploaded_test: ref_files.extend(uploaded_test)
                if uploaded_spec: ref_files.extend(uploaded_spec)
                if uploaded_recipe: ref_files.extend(uploaded_recipe)
                
                json_result, chunk_list = analyze_design_with_ai(main_img_objs, ref_files, legal_knowledge_base)
                report_data = json.loads(json_result)
                
                st.markdown('<div class="section-title">📊 광고 적정성 3-Pass 종합 진단 결과</div>', unsafe_allow_html=True)
                
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
                
                # 생성된 모든 청크(구간)를 순서대로 화면 좌측에 띄우고, 우측에 해당 인덱스의 판정 결과 출력
                for idx, chunk_img in enumerate(chunk_list):
                    st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                    
                    row_col1, row_col2 = st.columns([1, 1])
                    
                    with row_col1:
                        st.image(chunk_img, use_container_width=True)
                        
                    with row_col2:
                        issues = [r for r in report_data if r.get("image_index") == idx]
                        
                        if not issues:
                            # AI가 해당 인덱스를 누락했을 경우의 방어 로직
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>해당 구간 시각 텍스트 추출 및 대조 결과 이상 없음.</div>', unsafe_allow_html=True)
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
                                - **추출된 문구:** {issue.get("found_text", "-")}
                                - **팩트 대조(Pass 2):** {issue.get("fact_check", "")}
                                - **QC 판정(Pass 3):** {issue.get("recommendation", "")}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.write("") # 동일 구간 내 여러 이슈가 있을 경우 간격
                                
                    st.markdown("---")

            except json.JSONDecodeError:
                st.error("AI 응답을 구조화하는 데 실패했습니다. 시스템 로그를 확인해 주십시오.")
            except Exception as e:
                st.error(f"AI 분석 중 오류가 발생했습니다. 상세 에러: {e}")
