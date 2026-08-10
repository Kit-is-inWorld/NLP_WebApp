import streamlit as st
import pandas as pd
import json
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
from pythainlp.tag import pos_tag

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="Food & Product Review NLP Analyzer",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CSS Custom Styling (Warm Dark-Yellow / Amber Theme) -----------------
st.markdown("""
<style>
    /* Gradient Background - Warm Amber/Deep Yellow Theme */
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%) !important;
        font-family: 'Sarabun', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #fff9e6 !important;
        border-right: 2px solid #e0a926 !important;
    }
    
    /* Main Card Container */
    .main-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px rgba(180, 115, 0, 0.15);
        margin-bottom: 20px;
        border: 1px solid rgba(224, 169, 38, 0.3);
    }
    
    /* Prominent Input Header */
    .input-label-large {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #7c4a03 !important;
        margin-bottom: 8px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Custom Metric Box */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border-left: 6px solid #e67e22;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #7f8c8d;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 800;
        color: #2c3e50;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Badges */
    .tag-badge {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 3px;
    }
    
    /* Highlight Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #d35400 !important;
        border-bottom: 3px solid #d35400 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- NLP Initialization -----------------
stopwords = set(thai_stopwords())

# ----------------- NLP Pipeline Functions -----------------
def step1_clean_text(text: str) -> dict:
    """Regex Cleansing: ลบ URL, เบอร์โทร, อีโมจิ, แฮชแท็ก, คำลากเสียง, และ Whitespace"""
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
    """NER สกัดเอนทิตีอัจฉริยะ (Location, Date, Store/Brand, Price, Contact)"""
    entities = []
    
    # 1. Location Detection
    provinces = ["เชียงใหม่", "กรุงเทพ", "พัทยา", "ขอนแก่น", "ภูเก็ต", "อารีย์", "สยามสแควร์", "สยาม", "สุขุมวิท", "ลาดพร้าว", "หัวหิน", "ชลบุรี"]
    for prov in provinces:
        if prov in text:
            entities.append({"Entity": prov, "Type": "LOCATION (สถานที่/พิกัด)"})
            
    # 2. Store / Brand Detection (จากคีย์เวิร์ด 'ร้าน...')
    store_match = re.findall(r'(?:ร้าน|สาขา)\s*([ก-๙a-zA-Z0-9_\s]+?)(?=\s+(?:สาขา|แถว|ที่|วันที่|อาหาร|พนักงาน|จานละ|บรรยากาศ|$))', text)
    for s in store_match:
        s_clean = s.strip()
        if len(s_clean) > 2 and s_clean not in provinces:
            entities.append({"Entity": s_clean, "Type": "ORGANIZATION / STORE (ชื่อร้าน/แบรนด์)"})
            
    # 3. Date / Time
    date_matches = re.findall(r'(\d{1,2}\s*(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|\bม\.ค\.|\bก\.พ\.|\bมี\.ค\.))', text)
    for d in date_matches:
        entities.append({"Entity": d, "Type": "DATE_TIME (วัน/เวลา)"})
        
    # 4. Price Detection
    price_matches = re.findall(r'(\d+[\d,]*\s*บาท)', text)
    for p in price_matches:
        entities.append({"Entity": p, "Type": "PRICE (ราคา/มูลค่า)"})
        
    # Deduplicate entities
    unique_entities = []
    seen = set()
    for item in entities:
        key = (item["Entity"], item["Type"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(item)
            
    return unique_entities

# ----------------- UI Layout -----------------

# Header Banner
st.markdown("""
<div style="background: rgba(255,255,255,0.92); border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); border-bottom: 4px solid #e67e22;">
    <h1 style="color: #d35400; margin: 0; font-size: 36px; font-weight: 800;">🍲 Food & Product Review NLP Analyzer</h1>
    <p style="color: #7f8c8d; font-size: 16px; margin-top: 6px; margin-bottom: 0;">
        ระบบอัจฉริยะวิเคราะห์ข้อความรีวิวอาหารและสินค้า สกัดประเด็นสำคัญ (Aspect) ตัดสิ่งรบกวน (Cleansing) และระบุ Entities (NER)
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ ตัวอย่างข้อมูลทดสอบ (Preset Data)")
    
    preset_choice = st.selectbox(
        "เลือกตัวอย่างรีวิวเพื่อประเมินผล:",
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
    #### 📋 ขั้นตอนการประมวลผล:
    1. **Regex Cleansing:** ลบ Noise, URLs, Phones, Emojis
    2. **Tokenization:** ตัดคำไทย & กรอง Stopwords
    3. **Aspect Classification:** จำแนกหัวข้อหลัก
    4. **NER & POS Tagging:** สกัดเอนทิตีและชนิดคำ
    """)

# Main Content Box
default_val = preset_texts[preset_choice] if preset_choice != "✍️ กำหนดข้อความเอง (Custom Input)" else "ร้าน ปิ้งย่างบุฟเฟ่ต์ สาขา อารีย์ อาหารอร่อยมาก เนื้อพรีเมียม แต่ราคาแพงไปนิด โทร 081-444-5555 #อร่อย"

st.markdown('<div class="input-label-large">✍️ ป้อนข้อความรีวิวที่ต้องการวิเคราะห์:</div>', unsafe_allow_html=True)
user_text = st.text_area(
    label="Input Box",
    label_visibility="collapsed",
    value=default_val, 
    height=120,
    placeholder="พิมพ์หรือวางข้อความรีวิวอาหารและสินค้าที่นี่..."
)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    run_process = st.button("🚀 ประมวลผลข้อความ (Run NLP Pipeline)", type="primary", use_container_width=True)

if run_process:
    if not user_text.strip():
        st.warning("⚠️ กรุณาใส่ข้อความรีวิวก่อนกดประมวลผล")
    else:
        # Run Pipeline
        clean_res = step1_clean_text(user_text)
        cleaned_text = clean_res["cleaned"]
        raw_tokens, filtered_tokens = step2_normalize_and_tokenize(cleaned_text)
        aspect, matched_kws = step3_classify_aspect(filtered_tokens)
        entities_list = robust_ner_extractor(user_text)
        pos_tags = pos_tag(filtered_tokens, engine='perceptron')

        total_noises = sum(len(v) for v in clean_res['removed_elements'].values())

        # Executive Metrics Cards
        st.markdown("<h3 style='color: #7c4a03; margin-top: 20px;'>📊 ผลการวิเคราะห์สรุป (Executive Summary)</h3>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e74c3c;">
                <div class="metric-title">หมวดหมู่หลัก (Aspect Topic)</div>
                <div class="metric-value" title="{aspect}" style="color: #c0392b;">{aspect}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3498db;">
                <div class="metric-title">คำสำคัญ (Keywords)</div>
                <div class="metric-value" style="color: #2980b9;">{len(filtered_tokens)} คำ</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #2ecc71;">
                <div class="metric-title">Entity ที่พบ (NER)</div>
                <div class="metric-value" style="color: #27ae60;">{len(entities_list)} รายการ</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f39c12;">
                <div class="metric-title">Noise ที่ลบออก (Cleansing)</div>
                <div class="metric-value" style="color: #d35400;">{total_noises} รายการ</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tab Sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧹 1. Text Cleansing", 
            "🔤 2. Tokenization", 
            "🏷️ 3. Topic & Keywords", 
            "🔍 4. POS & NER"
        ])

        with tab1:
            st.markdown("#### การทำความสะอาดข้อความด้วย Regular Expressions (Regex)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**ข้อความเดิม (Raw Input):**")
                st.info(user_text)
            with c2:
                st.markdown("**ข้อความหลังคลีน (Cleaned Text):**")
                st.success(cleaned_text if cleaned_text else "(ว่างเปล่า)")

            st.markdown("##### 🗑️ ข้อมูล Noise ที่ถูกกรองออก:")
            st.json(clean_res["removed_elements"])

        with tab2:
            st.markdown("#### การตัดคำและการปรับรูปคำ (Normalization & Tokenization)")
            st.write(f"• **จำนวนคำทั้งหมดก่อนตัด Stopwords:** `{len(raw_tokens)}` คำ")
            st.write(f"• **จำนวนคำสำคัญที่มีความหมาย (Keywords):** `{len(filtered_tokens)}` คำ")
            
            st.markdown("**🏷️ คำสำคัญที่สกัดได้:**")
            badges_html = "".join([f'<span class="tag-badge">{word}</span>' for word in filtered_tokens])
            st.markdown(badges_html, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### ผลการจัดหมวดหมู่ประเด็น (Aspect Identification)")
            st.success(f"📌 **หมวดหมู่ที่ตรวจพบ:** {aspect}")
            
            if matched_kws:
                st.write("**คำสำคัญที่เป็นตัวชี้วัด (Trigger Keywords):**")
                for kw in matched_kws:
                    st.markdown(f"- 🎯 `{kw}`")
            else:
                st.info("ไม่พบคีย์เวิร์ดเฉพาะเจาะจง จัดเป็นข้อความทั่วไป")

        with tab4:
            st.markdown("#### การสกัด Named Entities (NER) และชนิดของคำ (POS)")
            col_ner, col_pos = st.columns(2)
            
            with col_ner:
                st.markdown("##### 📍 Named Entities ที่ตรวจพบ (NER)")
                if entities_list:
                    df_ner = pd.DataFrame(entities_list)
                    st.dataframe(df_ner, use_container_width=True, hide_index=True)
                else:
                    st.warning("ไม่พบ Named Entity ในข้อความนี้")
                    
            with col_pos:
                st.markdown("##### 🔠 Part-of-Speech Tags (ชนิดของคำ)")
                if pos_tags:
                    df_pos = pd.DataFrame(pos_tags, columns=["คำศัพท์ (Word)", "POS Tag"])
                    st.dataframe(df_pos, use_container_width=True, height=260)