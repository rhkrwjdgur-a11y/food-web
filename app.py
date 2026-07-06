import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2
import json
import time
import requests
import urllib.parse
from datetime import datetime
import re

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
    table { width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 10px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    th { background-color: #f2f2f2; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. API 키 설정
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
    if not os.path.exists(docs_path): return "", "문서 폴더 없음"
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
    if not pdf_files: return "", "PDF 없음"
    for filename in pdf_files:
        try:
            with open(os.path.join(docs_path, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i in range(min(len(reader.pages), 20)):
                    text = reader.pages[i].extract_text()
                    if text: knowledge_text += text + "\n"
        except Exception: pass
    return knowledge_text, None

# 3-2. 공공데이터포털 영양성분 DB 호출 함수
def query_food_nutrient_db(food_name):
    if not food_name: return None
    
    std_dict = {
        "쇠고기": "소고기",
        "계육": "닭고기",
        "돈육": "돼지고기"
    }
    search_name = std_dict.get(food_name.strip(), food_name.strip())
    
    if not FOOD_API_KEY or FOOD_API_KEY == "your_api_key_here":
        st.sidebar.error("API 키가 설정되지 않았습니다.")
        return None

    encoded_name = urllib.parse.quote(search_name)
    url = f"http://apis.data.go.kr/1471000/FoodNtrIrdntInfoService1/getFoodNtrItdntList1?ServiceKey={FOOD_API_KEY}&desc_kor={encoded_name}&pageNo=1&numOfRows=200&type=json"
    
    try:
        response = requests.get(url, timeout=15)
        res_text = response.text.strip()
        
        if not res_text or res_text.startswith('<'):
            return None
            
        res_json = json.loads(res_text)
        
        if 'cmmMsgHeader' in res_json or 'OpenAPI_ServiceResponse' in res_json:
            return None
        
        header = res_json.get('header') or res_json.get('response', {}).get('header', {})
        if header.get('resultCode') and header.get('resultCode') != '00':
            return None
            
        body = res_json.get('body') or res_json.get('response', {}).get('body', {})
        items = body.get('items', [])
        
        if not items:
            return []
            
        if isinstance(items, dict) and 'item' in items:
            return items['item']
        elif isinstance(items, list):
            return items
        else:
            return [items]
            
    except Exception: 
        pass
        
    return None

# 4-1. Auto Pre-Scan
def auto_extract_db_keywords_json(main_images):
    model = genai.GenerativeModel('gemini-2.5-flash')
    payload = []
    for img in main_images:
        w, h = img.size
        if w > 1000: img = img.resize((1000, int(h * (1000.0/w))), Image.LANCZOS)
        payload.append(img)
    prompt = """
    당신은 식품 상세페이지에서 외부 영양성분 DB 대조에 필요한 '검색어'와 '세부 조건'을 추출하는 AI입니다.
    이미지를 훑어보고 타 식품과 수치를 비교하는 인포그래픽이 있다면 JSON 객체를 출력하십시오.
    
    [핵심 룰]
    API 검색용 '대분류 명사(Key)'는 무조건 띄어쓰기가 없는 단일 명사(예: 소고기, 닭고기, 대두)로만 작성하십시오.
    세부 수식어(Value) 항목에 시안에 적힌 괄호 안 글씨를 모두 적으십시오.
    
    [예시] {"소고기": "한우, 등심 구운것", "닭고기": "구운것", "대두": "노란콩 말린것"}
    비교 자료가 없다면 오직 "NONE" 이라고만 출력하십시오.
    """
    payload.append(prompt)
    try:
        response = model.generate_content(payload, generation_config=genai.types.GenerationConfig(temperature=0.0))
        res_text = response.text.strip()
        if res_text == "NONE" or not res_text: return {}
        res_text = re.sub(r'```json\s*', '', res_text)
        res_text = re.sub(r'```\s*', '', res_text)
        return json.loads(res_text)
    except Exception: return {}

# 4-2. 실시간 AI 비전 분석 로직 [유의어 매칭 방어막 추가]
def analyze_design_with_ai(main_images, ref_files, master_fact_files, legal_text, db_context_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    current_date_str = datetime.now().strftime("%Y년 %m월 %d일")
    
    content_payload = []
    chunk_list = []
    
    for idx, img_obj in enumerate(main_images):
        w, h = img_obj.size
        if w > 2000: 
            img_obj = img_obj.resize((2000, int(h * (2000.0/w))), Image.LANCZOS)
        chunk_list.append(img_obj)
        content_payload.append(f"--- [시안 구간 인덱스: {idx}] ---")
        content_payload.append(img_obj)
            
    if ref_files:
        for ref in ref_files:
            try: content_payload.append(Image.open(ref))
            except: pass 
    master_fact_count = 0
    if master_fact_files:
        for fact in master_fact_files:
            try: content_payload.append(Image.open(fact)); master_fact_count += 1
            except: pass
                
    prompt = f"""
    당신은 엄격하면서도 유연한 품질관리(QC) 전문가입니다. 제공된 시안 조각을 검토하십시오.
    
    [식약처 법령 지식 베이스]
    {legal_text}
    
    [요약된 국가 공인 영양성분 DB 데이터]
    {db_context_text if db_context_text else "경고: 외부 DB 데이터가 존재하지 않습니다."}
    
    [필수 강제 체크리스트 - 스킵 절대 금지]
    
    🚨 0. [인덱스 시야 격리 원칙]: 
       - 각 인덱스의 분석 결과를 작성할 때, 반드시 해당 인덱스 이미지 안에 직접 보이는 팩트 기반으로만 분석하십시오.

    🌟 1. 마케팅 수사(Puffery) 주의 환기:
       - 주관적/감성적 마케팅 문구만 있을 경우 "수정 권고"로 띄우고 과대광고 소지 검토를 요망하십시오.

    🔥 2. DB 비교 수치 검증 (유의어 매칭 룰 적용) 및 표 생성 강제:
       - 시안에 '구운것', '말린것'이라 적혀있어도 DB에는 '구이', '건조'로 등록되어 있을 수 있습니다. 토씨 하나까지 똑같지 않더라도 **의미상 동일한 조리법/부위의 데이터**를 찾아 수치를 대조하십시오.
       - 타 식품 수치 비교 그래프가 존재하면, DB 수치를 대조하여 무조건 마크다운 표(Table)를 생성하십시오.
       - DB에 의미상 부합하는 원물이 도저히 없으면 표 안에 '검색 결과 내 일치 원물 없음'이라고 명시하십시오.

    🔥 3. 당류 법적 용어 엄격 구분:
       - 영양정보표 당류가 0.5g 이상이라면 'ZERO' 표기를 금지하고 '설탕 무첨가/무가당'으로 수정 권고하십시오.

    🔥 4. 제조공정도 배합 기만 방어 및 [모든 콩류 원산지 100% 국산 대조 룰]:
       - 시안에 특정 하위 원료만을 단독 사용한 것처럼 묘사했다면 기만으로 적발하십시오.
       - 시안에서 '국산콩'을 강조했을 경우, [팩시안] 원재료명을 샅샅이 뒤져 제품에 들어간 '모든 콩류(대두, 약콩 등)'의 원산지가 예외 없이 100% '국산'인지 교차 검증하십시오. 수입산이 있다면 "치명적 위반"으로 적발하십시오.

    🔥 5. 포장재질 법적 표준 명칭 규격화:
       - 포장재 명칭이 '멸균종이팩'으로 기재되어 있다면 '멸균팩'으로 수정하도록 "수정 권고" 조치하십시오.
    
    반드시 아래의 JSON 배열(Array) 형식으로만 응답하십시오.
    [
      {{
        "image_index": 구간 인덱스 번호 (0부터 시작),
        "risk_level": "치명적 위반" 또는 "수정 권고" 또는 "정상",
        "title": "검토 항목 요약",
        "marketing_text": "상세페이지 추출 원문",
        "fact_or_legal_ground": "팩시안, 식약처 DB 매칭 항목, 또는 법적 잣대",
        "discrepancy_analysis": "위반 분석 및 조치 사항"
      }}
    ]
    * 위반사항이나 표를 그릴 내용이 없다면 risk_level "정상" 객체를 반환하십시오.
    """
    content_payload.append(prompt)
    
    for attempt in range(3):
        try:
            response = model.generate_content(content_payload, generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"))
            return response.text, chunk_list
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                if attempt < 2: time.sleep(10); continue
            raise e

# ==========================================
# 왼쪽 사이드바
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader("0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 식약처 영양성분 DB 실시간 자동 연동 (비교광고 검증용)")
db_search_keyword = st.sidebar.text_input("상세페이지 내 비교 대상 식품명 입력", help="비워두면 AI가 자동으로 탐지합니다.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 팩트 체크용 증빙 서류 (다중 업로드)")
uploaded_master_fact = st.sidebar.file_uploader("4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_test = st.sidebar.file_uploader("1️⃣ 시험성적서 및 추가 근거 자료", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_spec = st.sidebar.file_uploader("2️⃣ 원료 한글라벨/스펙", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
uploaded_recipe = st.sidebar.file_uploader("3️⃣ 배합비/레시피 데이터", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 3-Pass 투트랙 + 식약처 DB 자동 정밀 심사", use_container_width=True)

# ==========================================
# 최상단
# ==========================================
st.title("🛡️ 식품 상세페이지 표시·광고 사전 통제 시스템")
st.markdown("---")

legal_knowledge_base, learn_error = load_guideline_knowledge()
if not learn_error and legal_knowledge_base:
    st.info("📚 식약처 부당광고 고시 및 영양표시 지침 실시간 학습 완료")

# ==========================================
# 메인 화면
# ==========================================
if not uploaded_main_images:
    st.warning("👈 왼쪽 메뉴에서 의미 단위(배경/흐름)로 캡처하신 상세페이지 시안 이미지를 순서대로 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
        
    if not trigger_api:
        st.info("좌측 하단의 심사 가동 버튼을 누르면 AI 분석이 시작됩니다.")
        for img in main_img_objs: st.image(img, use_container_width=True)
    else:
        final_db_context_text = "" 
        
        with st.spinner("🔍 1단계: 시안 내 식약처 DB 타겟을 자동 추출하고 있습니다..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            if auto_dict:
                for base_food, detail_cond in auto_dict.items():
                    st.sidebar.success(f"🤖 탐지 완료: [{base_food}] ➔ 타겟 조건: {detail_cond}")
                    db_data = query_food_nutrient_db(base_food)
                    
                    # [데이터 다이어트 로직 적용] 불필요한 비타민/미네랄 JSON 데이터를 날리고 텍스트 1줄로 압축
                    if db_data:
                        simplified_db = []
                        for row in db_data[:200]:
                            name = row.get('DESC_KOR', '이름없음')
                            protein = row.get('NUTR_CONT3', 'N/A')
                            if protein != 'N/A':
                                simplified_db.append(f"- 명칭: [{name}] | 단백질 함량: {protein}g")
                        
                        final_db_context_text += f"\n[검색어 '{base_food}' (타겟 조건: {detail_cond}) DB 요약 데이터]\n" + "\n".join(simplified_db) + "\n"
                        st.sidebar.info(f"✅ 식약처 DB '{base_food}' 핵심 데이터 추출 완료")
            else:
                st.sidebar.info("🔍 탐지된 비교광고 외부 DB 키워드 없음")

        with st.spinner("⚙️ 2단계: 3-Pass 투트랙 정밀 심사 가동 중 (데이터 다이어트 및 유의어 매칭 적용)..."):
            try:
                ref_files = []
                if uploaded_test: ref_files.extend(uploaded_test)
                if uploaded_spec: ref_files.extend(uploaded_spec)
                if uploaded_recipe: ref_files.extend(uploaded_recipe)
                
                json_result, chunk_list = analyze_design_with_ai(main_img_objs, ref_files, uploaded_master_fact, legal_knowledge_base, final_db_context_text)
                report_data = json.loads(json_result)
                
                st.markdown('<div class="section-title">📊 광고 적정성 종합 진단 결과</div>', unsafe_allow_html=True)
                
                critical_cnt = sum(1 for r in report_data if r.get("risk_level") == "치명적 위반")
                warning_cnt = sum(1 for r in report_data if r.get("risk_level") == "수정 권고")
                pass_cnt = sum(1 for r in report_data if r.get("risk_level") == "정상")
                
                stat_c1, stat_c2, stat_c3 = st.columns(3)
                with stat_c1: st.markdown(f'<div class="metric-box">🚨 치명적 위반 <br><span class="metric-num" style="color:#dc3545;">{critical_cnt}건</span></div>', unsafe_allow_html=True)
                with stat_c2: st.markdown(f'<div class="metric-box">⚠️ 수정 권고(리뷰 요망) <br><span class="metric-num" style="color:#f39c12;">{warning_cnt}건</span></div>', unsafe_allow_html=True)
                with stat_c3: st.markdown(f'<div class="metric-box">✅ 정상 구간 <br><span class="metric-num" style="color:#2ecc71;">{pass_cnt}건</span></div>', unsafe_allow_html=True)
                
                st.write("")
                
                for idx, chunk_img in enumerate(chunk_list):
                    st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                    row_col1, row_col2 = st.columns([1, 1])
                    
                    with row_col1:
                        st.image(chunk_img, use_container_width=True)
                        
                    with row_col2:
                        issues = [r for r in report_data if r.get("image_index") == idx]
                        if not issues:
                            st.markdown('<div class="risk-pass"><div class="card-title">✅ 검토 완료</div>해당 구간 범용 법적 테두리 및 팩트 확인 완료.</div>', unsafe_allow_html=True)
                        else:
                            for issue in issues:
                                risk = issue.get("risk_level", "정상")
                                css_class = "risk-critical" if risk == "치명적 위반" else "risk-warning" if risk == "수정 권고" else "risk-pass"
                                icon = "❌" if risk == "치명적 위반" else "⚠️" if risk == "수정 권고" else "✅"
                                
                                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                                st.markdown(f'<div class="card-title">{icon} {issue.get("title", "")}</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                - **상세페이지 상태:** {issue.get("marketing_text", "-")}
                                - **QC 대조 기준:** {issue.get("fact_or_legal_ground", "-")}
                                - **분석 및 조치:** {issue.get("discrepancy_analysis", "")}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.write("") 
                    st.markdown("---")

            except json.JSONDecodeError: st.error("AI 응답을 구조화하는 데 실패했습니다. 잠시 후 다시 시도해 주십시오.")
            except Exception as e: st.error(f"서버 과부하 오류가 발생했습니다. 잠시 대기 후 다시 시도해 주십시오. (에러: {e})")
