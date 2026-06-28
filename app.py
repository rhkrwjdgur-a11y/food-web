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
    .card-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
    .section-title { font-size: 20px; font-weight: bold; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 8px; margin-top: 25px; margin-bottom: 15px; }
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

# 4. 실시간 AI 비전 분석 로직 (JSON 강제 출력 및 좌표 추출 엔진)
def analyze_design_with_ai(image_obj, ref_files, legal_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_payload = []
    main_chunks = []
    
    # 메인 이미지 분할 (해상도 보존 팩트 로직)
    width, height = image_obj.size
    max_height = 10000
    
    if height > 16000:
        for i in range(0, height, max_height):
            box = (0, i, width, min(i + max_height, height))
            chunk = image_obj.crop(box)
            main_chunks.append(chunk)
            content_payload.append(chunk)
    else:
        main_chunks.append(image_obj)
        content_payload.append(image_obj)
        
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
    전송된 이미지 중 첫 {len(main_chunks)}장은 '메인 상세페이지 시안'을 상단부터 하단까지 순서대로 분할한 조각(Slice) 이미지들이며 (인덱스 0번부터 시작), 나머지는 팩트 체크를 위한 '증빙 서류'들입니다.
    
    [식약처 가이드라인 지식 베이스]
    {legal_text}
    
    다음 3가지 항목을 집중 검토하십시오:
    1. 원산지/원재료 거짓·과장 (메인 광고와 증빙 서류 팩트 불일치 여부)
    2. 영양강조표시 및 건강기능식품 오인 혼동 문구
    3. 예외조항 주석의 모호성
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오.
    [
      {{
        "slice_index": 위반이 발견된 메인 상세페이지 조각의 인덱스 번호 (0부터 시작하는 정수),
        "y_start_ratio": 해당 조각 내에서 위반 구역이 시작되는 세로 위치 비율 (0.0 ~ 1.0 소수점),
        "y_end_ratio": 해당 조각 내에서 위반 구역이 끝나는 세로 위치 비율 (0.0 ~ 1.0 소수점, 문구 위아래로 충분한 여백 포함),
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
    
    # 팩트: AI 응답을 무조건 JSON으로만 받도록 타입 강제 지정 (문자열 파싱 에러 방지)
    response = model.generate_content(
        content_payload,
        generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
    )
    
    return response.text, main_chunks

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

if not uploaded_image:
    st.warning("👈 왼쪽 메뉴에서 상세페이지 시안 이미지를 업로드해 주십시오.")
else:
    img = Image.open(uploaded_image)
    
    if trigger_api:
        with st.spinner("구글 Vision API 가동 중: 정밀 스캔 및 이미지 좌표 크롭 매칭을 진행하고 있습니다 (약 10~20초 소요)..."):
            try:
                ref_files = []
                if uploaded_test: ref_files.extend(uploaded_test)
                if uploaded_spec: ref_files.extend(uploaded_spec)
                if uploaded_recipe: ref_files.extend(uploaded_recipe)
                
                # AI 분석 호출 및 JSON 데이터 획득
                json_result, chunk_list = analyze_design_with_ai(img, ref_files, legal_knowledge_base)
                
                # 문자열로 반환된 JSON을 파이썬 딕셔너리로 완벽 변환
                report_data = json.loads(json_result)
                
                st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
                
                if not report_data:
                    st.success("✅ 심사 완료: 식약처 고시 위반 및 증빙 서류 불일치 리스크가 발견되지 않았습니다.")
                else:
                    # 요약 지표 계산
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
                    
                    # AI가 찾아낸 좌표를 바탕으로 이미지를 자동 크롭하여 결과와 나란히 배치
                    for idx, issue in enumerate(report_data):
                        slice_idx = issue.get("slice_index", 0)
                        
                        row_col1, row_col2 = st.columns([1, 2])
                        
                        with row_col1:
                            if slice_idx < len(chunk_list):
                                target_chunk = chunk_list[slice_idx]
                                cw, ch = target_chunk.size
                                sy = int(ch * issue.get("y_start_ratio", 0.0))
                                ey = int(ch * issue.get("y_end_ratio", 1.0))
                                
                                # 여백이 너무 좁을 경우를 대비한 보정
                                if ey <= sy: ey = min(sy + int(ch*0.1), ch)
                                
                                cropped_img = target_chunk.crop((0, sy, cw, ey))
                                st.image(cropped_img, caption=f"[AI 문제 인식 구역 {idx+1}]", use_container_width=True)
                        
                        with row_col2:
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
                        
                        st.write("---") # 항목 간 구분선

            except json.JSONDecodeError:
                st.error("AI 응답을 구조화하는 데 실패했습니다. 시스템 로그를 확인해 주십시오.")
            except Exception as e:
                st.error(f"이미지 크롭 또는 AI 분석 중 오류가 발생했습니다. 상세 에러: {e}")
