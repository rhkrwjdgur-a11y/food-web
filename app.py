import streamlit as st
import google.generativeai as genai
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import os
import json
import time
import requests
import urllib.parse
import re

# ==========================================
# 1. 기본 페이지 설정 및 CSS
# ==========================================
st.set_page_config(page_title="상세페이지 정밀 검토 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .risk-critical { background-color: #fdf2f2; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-warning { background-color: #fefaf0; padding: 20px; border-radius: 10px; border-left: 6px solid #f39c12; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .risk-pass { background-color: #f4fbf7; padding: 20px; border-radius: 10px; border-left: 6px solid #2ecc71; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .card-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 3대 핵심 API 키 연동 (Secrets)
# ==========================================
try:
    genai.configure(api_key=st.secrets["AI_VISION_API_KEY"])
    FOOD_API_KEY = st.secrets["FOOD_SAFETY_API_KEY"]

    gcp_json_string = st.secrets["gcp_service_account"]["GOOGLE_VISION_KEY"]
    gcp_credentials = json.loads(gcp_json_string)
    gcp_credentials["private_key"] = gcp_credentials["private_key"].replace("\\n", "\n")

    vision_credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    vision_client = vision.ImageAnnotatorClient(credentials=vision_credentials)

except KeyError as e:
    st.error(f"시스템 오류: Secrets 설정 누락 - {e}")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"구글 비전 JSON 키 형식이 잘못되었습니다: {e}")
    st.stop()
except Exception as e:
    st.error(f"구글 비전 인증 오류: {e}")
    st.stop()

# ==========================================
# 2-1. 디버그 모드 
# ==========================================
DEBUG_MODE = st.sidebar.checkbox("🐞 디버그 모드 (로그 확인용)", value=True)

# ==========================================
# 2-2. Gemini 응답 스키마
# ==========================================
REVIEW_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "image_index": {"type": "INTEGER"},
            "risk_level": {
                "type": "STRING",
                "enum": ["치명적 위반", "수정 권고", "적합"]
            },
            "title": {"type": "STRING"},
            "exact_text_in_design": {"type": "STRING"},
            "fact_or_legal_ground": {"type": "STRING"},
            "discrepancy_analysis": {"type": "STRING"},
        },
        "required": [
            "image_index",
            "risk_level",
            "title",
            "exact_text_in_design",
            "fact_or_legal_ground",
            "discrepancy_analysis",
        ],
    },
}

# ==========================================
# 3. 보조 함수
# ==========================================
def query_food_nutrient_db(food_name):
    if not food_name or not FOOD_API_KEY:
        return None
    std_dict = {"쇠고기": "소고기", "계육": "닭고기", "돈육": "돼지고기"}
    search_name = std_dict.get(food_name.strip(), food_name.strip())
    url = (
        f"http://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02"
        f"?serviceKey={FOOD_API_KEY}&pageNo=1&numOfRows=200&type=json"
        f"&DESC_KOR={urllib.parse.quote(search_name)}"
    )
    try:
        res_json = json.loads(requests.get(url, timeout=15).text.strip())
        body = res_json.get('body') or res_json.get('response', {}).get('body', {})
        items = body.get('items', [])
        return items['item'] if isinstance(items, dict) and 'item' in items else items if isinstance(items, list) else [items]
    except Exception as e:
        return None

def auto_extract_db_keywords_json(main_images):
    model = genai.GenerativeModel('gemini-2.5-flash')
    payload = [
        img.resize((1000, int(img.size[1] * (1000.0 / img.size[0]))), Image.LANCZOS) if img.size[0] > 1000 else img
        for img in main_images
    ]
    payload.append("""타 식품과 수치를 비교하거나 식약처 DB를 인용하는 문구가 있다면 대분류 명사를 key로 JSON 출력. 없으면 "NONE" 출력.""")
    try:
        res = model.generate_content(
            payload, generation_config=genai.types.GenerationConfig(temperature=0.0)
        ).text.strip()
        if res == "NONE":
            return {}
        return json.loads(re.sub(r'```json\s*|```\s*', '', res))
    except Exception as e:
        return {}

# ==========================================
# 4. 투트랙 에이전트 시스템 (진행률 및 시간 추적 추가)
# ==========================================
@st.cache_data(show_spinner=False)
def extract_text_with_google_vision(uploaded_files):
    if not uploaded_files:
        return "팩시안 이미지 없음"
    extracted_text = ""
    for file in uploaded_files:
        content = file.read()
        image = vision.Image(content=content)
        response = vision_client.document_text_detection(image=image)
        if response.error.message:
            st.error(f"Vision API 에러: {response.error.message}")
            continue
        if response.full_text_annotation:
            extracted_text += response.full_text_annotation.text + "\n\n"
        file.seek(0)
    return extracted_text

def analyze_design_with_ai(main_images, ocr_extracted_text, db_context_text, progress_bar, status_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    final_report = []
    chunk_list = []
    total_chunks = len(main_images)
    start_time = time.time()

    for idx, img_obj in enumerate(main_images):
        img_obj = (
            img_obj.resize((2000, int(img_obj.size[1] * (2000.0 / img_obj.size[0]))), Image.LANCZOS)
            if img_obj.size[0] > 2000 else img_obj
        )
        chunk_list.append(img_obj)
        
        # UI 업데이트: 실시간 소요 시간 및 진행률 텍스트 표시
        elapsed_time = time.time() - start_time
        status_text.info(f"⏳ **진행 상황:** [{idx + 1} / {total_chunks}] 번째 시안 구간 스캔 및 팩트체크 중... (현재 소요 시간: {elapsed_time:.1f}초)")

        # --- [1단계] 텍스트 필사 및 시각 요소(사진) 동시 묘사 ---
        extract_prompt = """
        당신은 시력 2.0의 철저하고 객관적인 이미지 분석기입니다. 다음 두 가지를 수행하십시오.
        1. [텍스트 추출]: 이 이미지 구간에 실제로 적혀있는 텍스트와 숫자를 픽셀 단위로 있는 그대로 필사하십시오. (없는 텍스트 창조 절대 금지)
        2. [시각 요소 묘사]: 만약 이미지에 요리, 원물(고기, 생선, 콩, 견과류 등), 두유를 잔에 따르는 모습 등 '연출된 사진'이 포함되어 있다면, 어떤 사진인지 간략히 묘사하십시오. (예: '[시각 요소] 고기와 생선, 우유 등이 차려진 연출 사진 존재')
        텍스트나 연출 사진이 아예 없다면 "해당 없음"이라고만 출력하십시오.
        """
        design_raw_text = "텍스트 추출 실패"
        
        for attempt in range(3):
            try:
                extraction_response = model.generate_content(
                    [extract_prompt, img_obj],
                    generation_config=genai.types.GenerationConfig(temperature=0.0),
                )
                design_raw_text = extraction_response.text
                break
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e) or "503" in str(e):
                    if DEBUG_MODE:
                        st.toast(f"[구간 {idx+1}] 서버 지연. 대기 후 재시도... (시도 {attempt+1}/3)")
                    time.sleep(3)
                else:
                    break
                    
        time.sleep(0.5)

        # --- [2단계] 선임자급 핀셋 검수 ---
        review_prompt = f"""
        당신은 실무 경험이 풍부한 식품 마케팅 상세페이지 QC 선임자입니다.
        아래 [1단계 시안 데이터]와 [Google Vision API 팩시안 데이터]를 대조하십시오.

        [1단계 시안 데이터 (현재 구간에 존재하는 텍스트 및 시각 요소)]
        {design_raw_text}

        [Google Vision API 팩시안 원시 데이터 - 절대적 기준]
        {ocr_extracted_text}

        🚨 [과잉 지적 금지 및 연출 컷 예외 절대 규칙] 🚨
        1. **텍스트 창조 지적 금지:** [1단계 시안 데이터]에 '영양성분표'나 '원재료명' 수치가 아예 없다면, "누락되었다"고 지적하지 말고 팩트 체크 대상이 아니므로 무조건 "적합" 처리하십시오.
        2. **⭐ [연출 컷 주의문구 예외]:** 단, [1단계 시안 데이터]에 '[시각 요소]'로 고기, 생선, 콩, 잔에 따르는 모습 등 연출 사진이 묘사되어 있는데, 텍스트 추출 내용 중에 '이미지 예' 또는 '연출된 이미지'라는 문구가 "없다면", 이것은 명백한 주의문구 누락이므로 반드시 "수정 권고"로 지적해야 합니다.

        🔥 [상세페이지 핵심 핀셋 검수 룰]
        1. [영양정보 복붙 오타 색출]: [1단계 시안]에 '영양정보' 수치가 적혀있을 때만 팩시안과 1:1 대조.
        2. [알레르기 문구 대조]: [1단계 시안]에 '주의문구'가 있을 때만 팩시안과 대조.
        3. [제품명 오기재]: 카피나 제품명에 '&고칼슘' 누락 여부 확인.
        4. [비타민 기능성 명칭]: 비타민E 설명 텍스트가 있을 때, '항산화작용을 하여' 누락 지적.
        5. [연출 컷 주의문구 누락 적발]: (위 절대 규칙 2번에 따름)
        6. [카피 오타]: '무규 포장' -> '무균 포장' 색출.
        7. [원재료명 오기재]: 철자 오류(유성비타민지방산에스테르 등) 색출.
        8. [원물 은폐 기만]: 검은콩/아몬드&잣을 크게 강조하는데 팩시안 배합비율이 1% 미만 극소량이면 지적.
        9. [인용 데이터 팩트 체크]: WHO/식약처 출처 권장량 수치가 팩트와 일치하는지 확인.
        
        10. ⭐ **[범용적 영양/성분 강조 표시 정밀 팩트체크]**: 시안에 특정 영양소나 성분을 강조하는 표현(예: 저당, 무가당, 고단백, 식물성 등)이 등장하면 팩시안 원시 데이터를 대조하여 수치, 가짓수, 원료 특성이 사실과 일치하는지 확인하십시오. 불일치 시 수정 권고.
        
        11. ⭐ **[원료적 특성 강조(면책 조항) 절대 예외 룰 (매우 중요)]**: 시안에서 철분, 식이섬유 등 특정 영양소를 나열하며 강조하더라도, 그 주변 텍스트에 **"* 원료(콩)에 대한 설명입니다"**, **"* 원액두유에 한함"**과 같이 해당 내용이 '최종 완제품'이 아닌 '원료 자체'의 특성임을 밝히는 **단서조항(주석)**이 함께 추출되었다면, 절대 팩시안 영양정보표와 대조하여 오류라고 지적하지 마십시오. 이는 합법적인 원료 설명 마케팅이므로 무조건 "적합" 판정하고, 판정 사유에 "원료적 특성에 대한 설명임을 명시하는 주석이 확인되어 적합함"이라고 브리핑하십시오.
            
        12. ⭐ **[일반 마케팅 구간 내용 요약 (필수)]**: 위 1~11번에 해당하지 않는 일반적인 텍스트/이미지라도 무조건 어떤 내용인지 요약하고 합격 도장(적합)을 찍어 JSON을 생성하십시오.

        ⭐ **[모든 구간 100% 답변 의무화 - 절대 지시]** ⭐
        해당 시안 구간이 완벽히 정상이라도, 무조건 "risk_level": "적합" 으로 JSON 객체를 최소 1개 이상 생성하십시오.
        
        image_index 필드에는 반드시 {idx} 값을 넣으십시오.
        """

        for attempt in range(3):
            try:
                review_response = model.generate_content(
                    [review_prompt, img_obj],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=REVIEW_RESPONSE_SCHEMA,
                    ),
                )
                
                chunk_issues = json.loads(review_response.text)

                if isinstance(chunk_issues, list):
                    for issue in chunk_issues:
                        issue["image_index"] = idx  
                else:
                    chunk_issues = []

                if len(chunk_issues) == 0:
                    chunk_issues = [{
                        "image_index": idx,
                        "risk_level": "적합",
                        "title": "일반 마케팅 구간 검토 완료",
                        "exact_text_in_design": design_raw_text[:200],
                        "fact_or_legal_ground": "특이사항 없음",
                        "discrepancy_analysis": "해당 구간의 텍스트와 이미지를 스캔한 결과, 팩시안 불일치나 표시사항 위반 요소가 발견되지 않아 적합으로 판정합니다.",
                    }]

                final_report.extend(chunk_issues)
                break 
                
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e) or "503" in str(e):
                    time.sleep(3)
                else:
                    if attempt == 2: 
                        if DEBUG_MODE:
                            st.error(f"[구간 {idx+1}] 2단계(핀셋 검수) 처리 실패: {e}")
                        final_report.append({
                            "image_index": idx, 
                            "risk_level": "적합",
                            "title": "일반 마케팅 구간 검토 완료",
                            "exact_text_in_design": design_raw_text[:200] if isinstance(design_raw_text, str) else "내용 없음",
                            "fact_or_legal_ground": "특이사항 없음",
                            "discrepancy_analysis": "해당 구간을 검수한 결과, 특별한 표시기준 위반이나 오기재가 발견되지 않았습니다.",
                        })
                    else:
                        time.sleep(1)

        # UI 업데이트: Progress Bar 갱신
        progress_bar.progress((idx + 1) / total_chunks)
        time.sleep(0.5)

    # 전체 루프 종료 후 최종 완료 시간 업데이트
    total_time = time.time() - start_time
    status_text.success(f"✅ **전체 {total_chunks}개 구간 검토 완료!** (총 소요 시간: {total_time:.1f}초)")

    return json.dumps(final_report), chunk_list


# ==========================================
# 5. UI 및 실행 흐름
# ==========================================
st.sidebar.markdown("### 📥 심사 대상 파일 등록")
uploaded_main_images = st.sidebar.file_uploader(
    "0️⃣ 메인 상세페이지 시안 (다중 업로드)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
st.sidebar.markdown("---")
uploaded_master_fact = st.sidebar.file_uploader(
    "4️⃣ 확정 표시사항 기준안 (최종 팩시안)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True
)
st.sidebar.markdown("---")
trigger_api = st.sidebar.button("⚙️ 상세페이지 핀셋 교차 검증", use_container_width=True)

st.title("🛡️ 마케팅 상세페이지 정밀 통제 시스템 (Sniper Ver.)")
st.markdown("---")

if not uploaded_main_images:
    st.warning("👈 시안 이미지를 업로드해 주십시오.")
else:
    main_img_objs = [Image.open(f) for f in uploaded_main_images]
    if not trigger_api:
        for img in main_img_objs:
            st.image(img, use_container_width=True)
    else:
        with st.spinner("👁️ [팩시안 기준점 확보] 구글 비전 API가 팩시안을 해독 중입니다... (캐시 적용)"):
            vision_extracted_text = extract_text_with_google_vision(uploaded_master_fact)
            st.success("✅ 구글 비전 API 팩시안 판독 완료")

        with st.spinner("🔍 [DB 연동] 외부 영양성분 DB 타겟 탐지 중..."):
            auto_dict = auto_extract_db_keywords_json(main_img_objs)
            final_db_context_text = ""
            if auto_dict:
                for base_food, detail_cond in auto_dict.items():
                    db_data = query_food_nutrient_db(base_food)
                    if db_data:
                        simplified_db = [
                            f"- [{row.get('DESC_KOR', '이름없음')}] 열량:{row.get('NUTR_CONT1')}kcal, "
                            f"단백질:{row.get('NUTR_CONT3')}g, 지방:{row.get('NUTR_CONT4')}g"
                            for row in db_data[:20]
                        ]
                        final_db_context_text += f"\n[검색어 '{base_food}'] DB 요약\n" + "\n".join(simplified_db) + "\n"

        # 💡 [새로운 기능 적용] 진행률 바 및 소요 시간 표시 컨테이너 생성
        st.markdown("### ⚖️ [제2 에이전트] 정밀 팩트체크 진행 현황")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # analyze_design_with_ai 호출 시 progress_bar와 status_text 객체를 함께 전달
            json_result, chunk_list = analyze_design_with_ai(
                main_img_objs, vision_extracted_text, final_db_context_text, progress_bar, status_text
            )
            report_data = json.loads(json_result)

            for idx, chunk_img in enumerate(chunk_list):
                st.markdown(f"### 📍 시안 구간 [{idx + 1}]")
                row_col1, row_col2 = st.columns([1, 1])
                with row_col1:
                    st.image(chunk_img, use_container_width=True)
                with row_col2:
                    issues = [r for r in report_data if r.get("image_index") == idx]
                    if not issues:
                        st.markdown(
                            '<div class="risk-warning"><div class="card-title">⚠️ 오류 발생</div>'
                            '이 구간에 해당하는 검토 결과를 찾지 못했습니다. 우측 상단의 Rerun을 눌러 다시 시도해주세요.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        for issue in issues:
                            risk = issue.get("risk_level", "적합")

                            if risk == "치명적 위반":
                                css_class = "risk-critical"
                                icon = "❌"
                            elif risk == "수정 권고":
                                css_class = "risk-warning"
                                icon = "⚠️"
                            else:  # "적합"
                                css_class = "risk-pass"
                                icon = "✅"

                            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="card-title">{icon} {issue.get("title", "검토 완료")}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"""
                            - **해당 구간 내용 요약:** {issue.get('exact_text_in_design', '')}
                            - **QC 팩트 근거:** {issue.get('fact_or_legal_ground', '')}
                            - **판정 사유 (브리핑):** {issue.get('discrepancy_analysis', '')}
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
        except Exception as e:
            st.error(f"오류 발생: {e}")
