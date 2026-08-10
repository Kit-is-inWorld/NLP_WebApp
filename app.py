import streamlit as st
import pandas as pd
import json
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
from pythainlp.tag import pos_tag

# Safe NER import เพื่อรองรับ PyThaiNLP ทุกเวอร์ชัน
ner_tagger = None
try:
    from pythainlp.tag.named_entity import NER
    ner_tagger = NER()
except Exception:
    try:
        from pythainlp.tag.named_entity import ThaiNameTagger
        ner_tagger = ThaiNameTagger()
    except Exception:
        ner_tagger = None

st.set_page_config(
    page_title="Food & Product Review NLP Analyzer",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

stopwords = set(thai_stopwords())

# ---------------- NLP Pipeline Functions ----------------
def step1_clean_text(text: str) -> dict:
    """Regex Cleansing: ลบ URL, เบอร์โทร, อีโมจิ, แฮชแท็ก, คำลากเสียง, และ Whitespace"""
    no_url = re.sub(r'https?://\S+|www\.\S+', '', text)
    no_phone = re.sub(r'(\+66|0)[0-9]{1,2}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '', no_url)
    no_hashtag = re.sub(r'#\S+', '', no_phone)
    no_emoji = re.sub(r'[\U00010000-\U0010ffff]', '', no_hashtag)
    no_repeated_chars = re.sub(r'(.)\1{2,}', r'\1', no_emoji)
    cleaned = re.sub(r'\s+', ' ', no_repeated_chars).strip()
    return {
        "cleaned": cleaned,
        "removed_elements": {
            "URLs": re.findall(r'https?://\S+|www\.\S+', text),
            "Phones": re.findall(r'(\+66|0)[0-9]{1,2}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text),
            "Hashtags": re.findall(r'#\S+', text)
        }
    }

def step2_normalize_and_tokenize(text: str):
    """Normalization & Word Tokenization with Stopwords removal"""
    normalized = normalize(text)
    raw_tokens = word_tokenize(normalized, engine='newmm')
    filtered_tokens = [w for w in raw_tokens if w.strip() and w not in stopwords]
    return raw_tokens, filtered_tokens

def step3_classify_aspect(tokens: list) -> tuple:
    """Aspect-based Topic Identification using Lexicon rules"""
    text_blob = "".join(tokens).lower()
    categories = {
        "รสชาติและคุณภาพอาหาร": ["อร่อย", "หวาน", "เค็ม", "เผ็ด", "จืด", "สด", "กรอบ", "นุ่ม", "คาว", "เหม็น", "ไม่อร่อย", "เข้มข้น", "แป้ง"],
        "บริการและพนักงาน": ["พนักงาน", "บริการ", "เสิร์ฟ", "พูดจา", "ช้า", "รอนาน", "เช็คบิล", "มารยาท", "หน้าบูด", "ผู้จัดการ"],
        "ราคาและความคุ้มค่า": ["แพง", "ถูก", "ราคา", "คุ้ม", "ปริมาณ", "จานเล็ก", "ลดราคา", "โปรโมชั่น", "บาท", "นิดเดียว"],
        "บรรยากาศและสถานที่": ["ที่จอดรถ", "ร้านสวย", "บรรยากาศ", "แอร์", "สะอาด", "สกปรก", "วิว", "สาขา", "ห้องน้ำ", "ร่มรื่น"]
    }
    
    scores = {}
    matched_kws = {}
    for cat, kws in categories.items():
        found = [kw for kw in kws if kw in text_blob]
        scores[cat] = len(found)
        matched_kws[cat] = found
        
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return "ทั่วไป / อื่นๆ", []
    return best_cat, matched_kws[best_cat]

def extract_entities_custom(tagged_list):
    """แปลงผลลัพธ์ BIO format เป็น Entity รายการ"""
    entities = []
    current_entity = []
    current_tag = None
    
    for item in tagged_list:
        word = item[0]
        tag = item[1] if len(item) > 1 else 'O'
        
        if tag.startswith('B-'):
            if current_entity:
                entities.append(("".join(current_entity), current_tag))
                current_entity = []
            current_entity.append(word)
            current_tag = tag[2:]
        elif tag.startswith('I-') and current_tag == tag[2:]:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(("".join(current_entity), current_tag))
                current_entity = []
                current_tag = None
    if current_entity:
        entities.append(("".join(current_entity), current_tag))
    return entities

def step4_extract_entities_and_pos(tokens: list, cleaned_text: str):
    """POS Tagging and Named Entity Recognition (NER)"""
    pos_tags = pos_tag(tokens, engine='perceptron')
    ner_entities = []
    if ner_tagger is not None:
        try:
            if hasattr(ner_tagger, 'get_ner'):
                ner_raw = ner_tagger.get_ner(cleaned_text)
                ner_entities = extract_entities_custom(ner_raw)
            elif hasattr(ner_tagger, 'tag'):
                ner_raw = ner_tagger.tag(cleaned_text)
                ner_entities = extract_entities_custom(ner_raw)
        except Exception:
            ner_entities = []
    return pos_tags, ner_entities

# ----------------- UI Layout -----------------
st.title("🍲 Food & Product Review NLP Analyzer")
st.markdown("**ระบบอัจฉริยะสำหรับคัดกรอง สกัดข้อมูลสำคัญ และจัดหมวดหมู่ข้อความรีวิวอาหารและสินค้า**")
st.divider()

# Sidebar: ตัวอย่างข้อมูลและเกี่ยวกับระบบ
with st.sidebar:
    st.header("⚙️ ข้อมูลทดสอบ (Preset Data)")
    
    preset_choice = st.selectbox(
        "เลือกตัวอย่างรีวิวเพื่อทดสอบ:",
        [
            "ตัวอย่างที่ 1: รสชาติและคุณภาพ (สยามเบเกอรี่)",
            "ตัวอย่างที่ 2: บริการและพนักงาน (ชาบูมาสเตอร์)",
            "ตัวอย่างที่ 3: ราคาและความคุ้มค่า (สเต็กเฮ้าส์)",
            "ตัวอย่างที่ 4: บรรยากาศและสถานที่ (กรีนลีฟ คาเฟ่)",
            "กำหนดข้อความเอง (Custom Input)"
        ]
    )
    
    preset_texts = {
        "ตัวอย่างที่ 1: รสชาติและคุณภาพ (สยามเบเกอรี่)": "ไปทานที่ร้าน สยามเบเกอรี่ สาขา เชียงใหม่ วันที่ 5 มกราคม เค้กช็อกโกแลตเข้มข้นอร่อยมากกกก 🍫🍰 แป้งนุ่มสดใหม่สุดๆ โทรสั่งล่วงหน้าได้ที่ 089-123-4567 เว็บไซต์ https://siambakery.th #รีวิวของกิน #อร่อยบอกต่อ",
        "ตัวอย่างที่ 2: บริการและพนักงาน (ชาบูมาสเตอร์)": "ร้าน ชาบูมาสเตอร์ แถว สยามสแควร์ พนักงานหน้าบูดมากกก พูดจาไม่มีหางเสียง 😡 สั่งน้ำซุปไปรอเกือบ 40 นาที ติดต่อผู้จัดการด่วน 02-999-8888 www.shabumaster.com #บริการแย่",
        "ตัวอย่างที่ 3: ราคาและความคุ้มค่า (สเต็กเฮ้าส์)": "กิน สเต็กเฮ้าส์ สาขา พัทยา จานละ 890 บาทแต่ได้เนื้อชิ้นนิดเดียว แพงเกินไปปปป 💸 ไม่สมราคาเลย ปริมาณน้อยมาก โทร 038-111-222 #รีวิวพัทยา",
        "ตัวอย่างที่ 4: บรรยากาศและสถานที่ (กรีนลีฟ คาเฟ่)": "ร้าน กรีนลีฟ คาเฟ่ ที่ ขอนแก่น บรรยากาศร่มรื่น ร้านสวยสะอาดมากกก 🌿 แอร์เย็น ที่จอดรถกว้างขวาง เหมาะมานั่งทำงาน https://greenleaf.cafe #คาเฟ่ขอนแก่น",
        "กำหนดข้อความเอง (Custom Input)": ""
    }
    
    st.markdown("---")
    st.info("💡 **ขั้นตอนการประมวลผล:**\n1. Regex Cleansing\n2. Tokenization & Normalization\n3. Topic/Aspect Classification\n4. POS Tagging & NER")

default_val = preset_texts[preset_choice] if preset_choice != "กำหนดข้อความเอง (Custom Input)" else "ร้าน ปิ้งย่างบุฟเฟ่ต์ สาขา อารีย์ อาหารอร่อยมาก เนื้อพรีเมียม แต่ราคาแพงไปนิด โทร 081-444-5555 #อร่อย"
user_text = st.text_area("✍️ ป้อนข้อความรีวิว:", value=default_val, height=120)

if st.button("🚀 ประมวลผลข้อความ (Run NLP Pipeline)", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("⚠️ กรุณาใส่ข้อความรีวิวก่อนกดประมวลผล")
    else:
        # Pipeline Processing
        clean_res = step1_clean_text(user_text)
        cleaned_text = clean_res["cleaned"]
        raw_tokens, filtered_tokens = step2_normalize_and_tokenize(cleaned_text)
        aspect, matched_kws = step3_classify_aspect(filtered_tokens)
        pos_tags, ner_entities = step4_extract_entities_and_pos(filtered_tokens, cleaned_text)

        # Dashboard Metrics
        st.subheader("📊 ผลการวิเคราะห์สรุป (Executive Summary)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("หมวดหมู่หลัก (Aspect Topic)", aspect)
        m2.metric("จำนวนคำสำคัญ (Keywords)", f"{len(filtered_tokens)} คำ")
        m3.metric("Entity ที่พบ (NER)", f"{len(ner_entities)} ตัว")
        m4.metric("Noise ที่ลบออก (Cleansing)", f"{sum(len(v) for v in clean_res['removed_elements'].values())} รายการ")

        st.markdown("---")

        # รายละเอียดแต่ละ Step
        t1, t2, t3, t4 = st.tabs(["🧹 1. Text Cleansing", "🔤 2. Tokenization", "🏷️ 3. Topic & Keywords", "🔍 4. POS & NER"])

        with t1:
            st.markdown("#### การทำความสะอาดข้อความด้วย Regex")
            c1, c2 = st.columns(2)
            c1.text_area("ข้อความเดิม (Raw Input):", user_text, height=100, disabled=True)
            c2.text_area("ข้อความหลังคลีน (Cleaned Text):", cleaned_text, height=100, disabled=True)
            
            st.write("**สิ่งที่ระบบลบออก:**")
            st.json(clean_res["removed_elements"])

        with t2:
            st.markdown("#### การปรับรูปคำและการตัดคำ (Normalization & Tokenization)")
            st.write(f"**จำนวนคำทั้งหมดก่อนตัด Stopwords:** {len(raw_tokens)} คำ")
            st.write(f"**คำสำคัญหลังตัด Stopwords ({len(filtered_tokens)} คำ):**")
            st.success(" | ".join(filtered_tokens))

        with t3:
            st.markdown("#### ผลการจัดกลุ่มประเด็นรีวิว (Aspect Identification)")
            st.info(f"📌 ข้อความนี้ถูกจัดอยู่ในหมวด: **{aspect}**")
            st.write("**คำสำคัญที่ใช้ในการตัดสินใจ (Trigger Keywords):**", matched_kws if matched_kws else "ไม่มี")

        with t4:
            st.markdown("#### การสกัดข้อมูลเฉพาะ (Named Entities) และชนิดของคำ (POS)")
            col_ner, col_pos = st.columns(2)
            
            with col_ner:
                st.markdown("**1. Named Entities ที่สกัดได้ (NER):**")
                if ner_entities:
                    df_ner = pd.DataFrame(ner_entities, columns=["Entity Text", "Label"])
                    st.dataframe(df_ner, use_container_width=True)
                else:
                    st.write("ไม่พบ Entity ในประโยค")
                    
            with col_pos:
                st.markdown("**2. ชนิดของคำ (POS Tags):**")
                df_pos = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                st.dataframe(df_pos, use_container_width=True, height=250)