import streamlit as st
import pandas as pd
import json
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
from pythainlp.tag import pos_tag

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Food & Product Review NLP Analyzer",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Clean & Modern CSS Styling -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@400;500;600;700&family=Sarabun:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Sarabun', sans-serif !important;
        background-color: #fbf8f1 !important;
        color: #2c3e50 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Kanit', sans-serif !important;
        letter-spacing: 0.3px;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #f4ede0 !important;
        border-right: 1px solid #e2d3b8 !important;
    }

    /* Header Banner Container */
    .app-header {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(180, 83, 9, 0.15);
        margin-bottom: 24px;
    }
    .app-header h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    .app-header p {
        font-size: 15px;
        margin: 6px 0 0 0;
        opacity: 0.92;
        font-weight: 300;
    }

    /* Section Card */
    .content-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #ebdcc5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    /* Input Field Label */
    .custom-input-label {
        font-family: 'Kanit', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #92400e;
        margin-bottom: 8px;
        display: block;
    }

    /* Metrics Card Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-item {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #f0e6d6;
        border-top: 4px solid #d97706;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        text-align: center;
    }
    .metric-item .title {
        font-size: 13px;
        font-weight: 600;
        color: #78716c;
        margin-bottom: 6px;
        font-family: 'Kanit', sans-serif;
    }
    .metric-item .val {
        font-size: 20px;
        font-weight: 700;
        color: #1c1917;
        font-family: 'Kanit', sans-serif;
    }

    /* Tag Badges */
    .tag-badge {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        margin: 4px 4px 4px 0;
    }

    /* Tabs Clean Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 2px solid #e7dfd1;
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Kanit', sans-serif !important;
        font-size: 15px;
        font-weight: 500;
        color: #78716c;
        padding: 10px 16px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #b45309 !important;
        font-weight: 700;
        border-bottom: 3px solid #b45309 !important;
    }

    /* Primary Button */
    div.stButton > button:first-child {
        background-color: #d97706 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-family: 'Kanit', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(217, 119, 6, 0.25) !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #b45309 !important;
        box-shadow: 0 5px 15px rgba(180, 83, 9, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- NLP Core Initialization -----------------
stopwords = set(thai_stopwords())

def step1_clean_text(text: str) -> dict:
    """Regex Cleansing: URLs, เบอร์โทร, อีโมจิ, แฮชแท็ก, การลากเสียง"""
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    phones = re.findall(r'(\+66|0)[0-9]{1,2}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text)
    hashtags = re.findall(r'#\S+', text)
    
    no_url = re.sub(r'https?://\S+|www\.\S+', '', text)
    no_phone = re.sub(r'(\+66|0)[0-9]{1,2}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '', no_url)
    no_hashtag = re.sub(r'#\S+', '', no_phone)
    no_emoji = re.sub(r'[\U00010000-\U0010ffff]', '', no_hashtag)
    no_repeated_chars = re.sub(r'(.)\1{2,}', r'\1', no_emoji)
    cleaned = re.sub(r'\s+', ' ', no_repeated_chars).strip()
    
    return {
        "cleaned": cleaned,
        "removed_elements": {
            "URLs": urls,
            "Phones": [p[0] if isinstance(p, tuple) else p for p in phones],
            "Hashtags": hashtags
        }
    }

def step2_normalize_and_tokenize(text: str):
    """Normalization & Word Tokenization with Stopwords removal"""
    normalized = normalize(text)
    raw_tokens = word_tokenize(normalized, engine='newmm')
    filtered_tokens = [w for w in raw_tokens if w.strip() and w not in stopwords and len(w) > 1]
    return raw_tokens, filtered_tokens

def step3_classify_aspect(tokens: list) -> tuple:
    """Aspect-based Topic Identification using Lexicon rules"""
    text_blob = "".join(tokens).lower()
    categories = {
        "รสชาติและคุณภาพอาหาร": ["อร่อย", "หวาน", "เค็ม", "เผ็ด", "จืด", "สด", "กรอบ", "นุ่ม", "คาว", "เหม็น", "ไม่อร่อย", "เข้มข้น", "แป้ง", "เนื้อ", "ช็อกโกแลต", "เค้ก", "ซุป"],
        "บริการและพนักงาน": ["พนักงาน", "บริการ", "เสิร์ฟ", "พูดจา", "ช้า", "รอนาน", "เช็คบิล", "มารยาท", "หน้าบูด", "ผู้จัดการ", "ต้อนรับ"],
        "ราคาและความคุ้มค่า": ["แพง", "ถูก", "ราคา", "คุ้ม", "ปริมาณ", "จานเล็ก", "ลดราคา", "โปรโมชั่น", "บาท", "นิดเดียว", "สมราคา"],
        "บรรยากาศและสถานที่": ["ที่จอดรถ", "ร้านสวย", "บรรยากาศ", "แอร์", "สะอาด", "สกปรก", "วิว", "สาขา", "ห้องน้ำ", "ร่มรื่น", "ทำงาน", "โต๊ะ"]
    }
    
    scores = {}
    matched_kws = {}
    for cat, kws in categories.items():
        found = [kw for kw in kws if kw in text_blob]
        scores[cat] = len(found)
        matched_kws[cat] = found
        
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return "ทั่วไป / ข้อเสนอแนะอื่นๆ", []
    return best_cat, matched_kws[best_cat]

def robust_ner_extractor(text: str):
    """NER สกัดเอนทิตีอัจฉริยะ (Location, Date, Store/Brand, Price)"""
    entities = []
    
    provinces = ["เชียงใหม่", "กรุงเทพ", "พัทยา", "ขอนแก่น", "ภูเก็ต", "อารีย์", "สยามสแควร์", "สยาม", "สุขุมวิท", "ลาดพร้าว", "หัวหิน", "ชลบุรี"]
    for prov in provinces:
        if prov in text:
            entities.append({"Entity": prov, "Type": "LOCATION (สถานที่/พิกัด)"})
            
    store_match = re.findall(r'(?:ร้าน|สาขา)\s*([ก-๙a-zA-Z0-9_\s]+?)(?=\s+(?:สาขา|แถว|ที่|วันที่|อาหาร|พนักงาน|จานละ|บรรยากาศ|$))', text)
    for s in store_match:
        s_clean = s.strip()
        if len(s_clean) > 2 and s_clean not in provinces:
            entities.append({"Entity": s_clean, "Type": "ORGANIZATION (ชื่อร้าน/แบรนด์)"})
            
    date_matches = re.findall(r'(\d{1,2}\s*(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|\bม\.ค\.|\bก\.พ\.|\bมี\.ค\.))', text)
    for d in date_matches:
        entities.append({"Entity": d, "Type": "DATE_TIME (วัน/เวลา)"})
        
    price_matches = re.findall(r'(\d+[\d,]*\s*บาท)', text)
    for p in price_matches:
        entities.append({"Entity": p, "Type": "PRICE (ราคา/มูลค่า)"})
        
    unique_entities = []
    seen = set()
    for item in entities:
        key = (item["Entity"], item["Type"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(item)
            
    return unique_entities

# ----------------- UI Content -----------------

# Header Banner
st.markdown("""
<div class="app-header">
    <h1>🍲 Food & Product Review NLP Analyzer</h1>
    <p>ระบบวิเคราะห์และคัดกรองข้อมูลรีวิวภาษาไทย: ขจัด Noise, สกัดคำสำคัญ (Keywords), ระบุเอนทิตี (NER) และจัดกลุ่มประเด็น (Aspects)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ ตัวเลือกข้อมูลทดสอบ")
    preset_choice = st.selectbox(
        "เลือกตัวอย่างรีวิวสำหรับการประเมิน:",
        [
            "ตัวอย่างที่ 1: รสชาติและคุณภาพ (สยามเบเกอรี่)",
            "ตัวอย่างที่ 2: บริการและพนักงาน (ชาบูมาสเตอร์)",
            "ตัวอย่างที่ 3: ราคาและความคุ้มค่า (สเต็กเฮ้าส์)",
            "ตัวอย่างที่ 4: บรรยากาศและสถานที่ (กรีนลีฟ คาเฟ่)",
            "✍️ กำหนดข้อความเอง (Custom Input)"
        ]
    )
    
    preset_texts = {
        "ตัวอย่างที่ 1: รสชาติและคุณภาพ (สยามเบเกอรี่)": "ไปทานที่ร้าน สยามเบเกอรี่ สาขา เชียงใหม่ วันที่ 5 มกราคม เค้กช็อกโกแลตเข้มข้นอร่อยมากกกก 🍫🍰 แป้งนุ่มสดใหม่สุดๆ โทรสั่งล่วงหน้าได้ที่ 089-123-4567 เว็บไซต์ https://siambakery.th #รีวิวของกิน #อร่อยบอกต่อ",
        "ตัวอย่างที่ 2: บริการและพนักงาน (ชาบูมาสเตอร์)": "ร้าน ชาบูมาสเตอร์ แถว สยามสแควร์ พนักงานหน้าบูดมากกก พูดจาไม่มีหางเสียง 😡 สั่งน้ำซุปไปรอเกือบ 40 นาที ติดต่อผู้จัดการด่วน 02-999-8888 www.shabumaster.com #บริการแย่",
        "ตัวอย่างที่ 3: ราคาและความคุ้มค่า (สเต็กเฮ้าส์)": "กิน สเต็กเฮ้าส์ สาขา พัทยา จานละ 890 บาทแต่ได้เนื้อชิ้นนิดเดียว แพงเกินไปปปป 💸 ไม่สมราคาเลย ปริมาณน้อยมาก โทร 038-111-222 #รีวิวพัทยา",
        "ตัวอย่างที่ 4: บรรยากาศและสถานที่ (กรีนลีฟ คาเฟ่)": "ร้าน กรีนลีฟ คาเฟ่ ที่ ขอนแก่น บรรยากาศร่มรื่น ร้านสวยสะอาดมากกก 🌿 แอร์เย็น ที่จอดรถกว้างขวาง เหมาะมานั่งทำงาน https://greenleaf.cafe #คาเฟ่ขอนแก่น",
        "✍️ กำหนดข้อความเอง (Custom Input)": ""
    }
    
    st.markdown("---")
    st.markdown("""
    **📌 NLP Pipeline Architecture:**
    1. **Regex Cleansing:** กรองสิ่งรบกวน
    2. **Tokenization:** ตัดคำไทย & ลบคำหยุด
    3. **Aspect Class:** จำแนกหัวข้อรีวิว
    4. **NER & POS:** สกัดเอนทิตี & ไวยากรณ์
    """)

# Input Card
default_val = preset_texts[preset_choice] if preset_choice != "✍️ กำหนดข้อความเอง (Custom Input)" else "ร้าน ปิ้งย่างบุฟเฟ่ต์ สาขา อารีย์ อาหารอร่อยมาก เนื้อพรีเมียม แต่ราคาแพงไปนิด โทร 081-444-5555 #อร่อย"

st.markdown('<div class="custom-input-label">✍️ ข้อความรีวิวที่ต้องการวิเคราะห์</div>', unsafe_allow_html=True)
user_text = st.text_area(
    label="Input Box",
    label_visibility="collapsed",
    value=default_val, 
    height=110,
    placeholder="พิมพ์หรือวางข้อความรีวิวที่นี่..."
)

btn_col1, _ = st.columns([1, 3])
with btn_col1:
    run_process = st.button("🚀 ประมวลผลข้อความ (Run NLP)", type="primary", use_container_width=True)

if run_process or user_text:
    if not user_text.strip():
        st.warning("⚠️ กรุณาใส่ข้อความรีวิวก่อนกดประมวลผล")
    else:
        # Pipeline Processing
        clean_res = step1_clean_text(user_text)
        cleaned_text = clean_res["cleaned"]
        raw_tokens, filtered_tokens = step2_normalize_and_tokenize(cleaned_text)
        aspect, matched_kws = step3_classify_aspect(filtered_tokens)
        entities_list = robust_ner_extractor(user_text)
        pos_tags = pos_tag(filtered_tokens, engine='perceptron')
        total_noises = sum(len(v) for v in clean_res['removed_elements'].values())

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 สรุปผลการวิเคราะห์ (Executive Summary)")

        # Metric Cards Layout
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-item" style="border-top-color: #e11d48;">
                <div class="title">หมวดหมู่หลัก (Aspect Topic)</div>
                <div class="val" style="color: #be123c;">{aspect}</div>
            </div>
            <div class="metric-item" style="border-top-color: #2563eb;">
                <div class="title">คำสำคัญ (Keywords)</div>
                <div class="val" style="color: #1d4ed8;">{len(filtered_tokens)} คำ</div>
            </div>
            <div class="metric-item" style="border-top-color: #16a34a;">
                <div class="title">Entity ที่พบ (NER)</div>
                <div class="val" style="color: #15803d;">{len(entities_list)} รายการ</div>
            </div>
            <div class="metric-item" style="border-top-color: #d97706;">
                <div class="title">Noise ที่ถูกลบ (Cleansing)</div>
                <div class="val" style="color: #b45309;">{total_noises} จุด</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs Layout
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧹 1. Text Cleansing", 
            "🔤 2. Tokenization", 
            "🏷️ 3. Aspect Classification", 
            "🔍 4. NER & POS Tagging"
        ])

        with tab1:
            st.markdown("##### เปรียบเทียบข้อความก่อนและหลังทำความสะอาดด้วย Regex")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("ข้อความเดิม (Raw Input):")
                st.info(user_text)
            with c2:
                st.caption("ข้อความหลังคลีน (Cleaned Text):")
                st.success(cleaned_text if cleaned_text else "(ว่างเปล่า)")

            st.markdown("##### รายละเอียด Noise ที่ตัดออก:")
            st.json(clean_res["removed_elements"])

        with tab2:
            st.markdown("##### ผลการปรับรูปคำและตัดคำ (Normalization & Tokenization)")
            st.write(f"• **จำนวนคำก่อนตัดคำหยุด:** `{len(raw_tokens)}` คำ")
            st.write(f"• **จำนวนคำสำคัญ (Keywords) หลังตัด Stopwords:** `{len(filtered_tokens)}` คำ")
            
            st.markdown("##### รายการคำสำคัญ:")
            badges_html = "".join([f'<span class="tag-badge">{w}</span>' for w in filtered_tokens])
            st.markdown(badges_html, unsafe_allow_html=True)

        with tab3:
            st.markdown("##### การจัดหมวดหมู่ประเด็น (Aspect-Based Topic)")
            st.info(f"📌 **หมวดหมู่ที่จำแนกได้:** **{aspect}**")
            
            if matched_kws:
                st.write("**คำสำคัญที่เป็นตัวกำหนดทิศทาง (Trigger Keywords):**")
                for kw in matched_kws:
                    st.markdown(f"- 🎯 `{kw}`")
            else:
                st.write("ไม่พบคีย์เวิร์ดบ่งชี้เฉพาะ ระบบจัดเป็นข้อความทั่วไป")

        with tab4:
            st.markdown("##### Named Entities (NER) และ Part-of-Speech (POS)")
            col_ner, col_pos = st.columns(2)
            
            with col_ner:
                st.markdown("**📍 Named Entities ที่ตรวจพบ:**")
                if entities_list:
                    df_ner = pd.DataFrame(entities_list)
                    st.dataframe(df_ner, use_container_width=True, hide_index=True)
                else:
                    st.info("ไม่พบ Entity ในข้อความ")
                    
            with col_pos:
                st.markdown("**🔠 ชนิดของคำ (POS Tags):**")
                if pos_tags:
                    df_pos = pd.DataFrame(pos_tags, columns=["คำศัพท์ (Word)", "POS Tag"])
                    st.dataframe(df_pos, use_container_width=True, height=260)