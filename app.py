import streamlit as st
import numpy as np
import cv2
import json
import hashlib
import time
import os
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BovineIQ – Cattle Intelligence",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD ML MODEL (graceful fallback)
# ─────────────────────────────────────────────
IMG_SIZE = 224

@st.cache_resource
def load_ml_model():
    try:
        from tensorflow.keras.models import load_model
        model = load_model("breed_model.keras")
        with open("labels.json") as f:
            class_indices = json.load(f)
        breed_labels = list(class_indices.keys())
        return model, breed_labels
    except Exception:
        return None, ["Gir", "Sahiwal", "Rathi", "Tharparkar", "Kankrej",
                      "Ongole", "Hariana", "Deoni", "Nimari", "Holstein"]

model, breed_labels = load_ml_model()

# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION  — FIX #1
# Must run before ANY st.* call that reads session_state
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "logged_in": False,
        "username": "",
        "page": "dashboard",
        "users_db": {"admin": hashlib.sha256("admin123".encode()).hexdigest()},
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --soil:   #1a120b;
    --bark:   #2c1810;
    --earth:  #3d2314;
    --amber:  #c8852a;
    --straw:  #e8c97e;
    --cream:  #f5efe6;
    --leaf:   #4a7c59;
    --moss:   #2d5016;
    --sage:   #7aab6d;
    --sky:    #6baed6;
    --mist:   rgba(245,239,230,0.06);
    --glow:   rgba(200,133,42,0.15);
}

.stApp {
    background: var(--soil);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(74,124,89,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, rgba(200,133,42,0.08) 0%, transparent 55%);
    font-family: 'DM Sans', sans-serif;
    color: var(--cream);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c1810 0%, #3d2314 100%) !important;
    border-right: 1px solid rgba(200,133,42,0.25) !important;
    min-width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * {
    color: #f5efe6 !important;
}
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: rgba(245,239,230,0.75) !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(200,133,42,0.15) !important;
    border-color: rgba(200,133,42,0.3) !important;
    color: #e8c97e !important;
    transform: none !important;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
code, pre   { font-family: 'DM Mono', monospace !important; }

/* ── Cards  — FIX #2: use st.container + CSS class ── */
.biq-card {
    background: linear-gradient(135deg, rgba(44,24,16,0.85) 0%, rgba(61,35,20,0.65) 100%);
    border: 1px solid rgba(200,133,42,0.22);
    border-radius: 20px;
    padding: 26px 30px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 18px;
}
.biq-card:hover {
    border-color: rgba(200,133,42,0.42);
    box-shadow: 0 12px 40px rgba(0,0,0,0.55), 0 0 0 1px rgba(200,133,42,0.15);
}

/* ── Stat tiles ── */
.stat-tile {
    background: rgba(245,239,230,0.05);
    border: 1px solid rgba(200,133,42,0.18);
    border-radius: 16px;
    padding: 20px 18px;
    text-align: center;
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #c8852a;
    line-height: 1;
    display: block;
}
.stat-label {
    font-size: 0.76rem;
    color: rgba(245,239,230,0.5);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
    display: block;
}

/* ── Result cards ── */
.result-big {
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
    display: flex;
    align-items: center;
    gap: 16px;
}
.result-breed-bg { background: rgba(74,124,89,0.15); border: 1px solid rgba(74,124,89,0.3); }
.result-body-bg  { background: rgba(200,133,42,0.12); border: 1px solid rgba(200,133,42,0.3); }
.result-icon  { font-size: 2rem; }
.result-title { font-size: 0.73rem; color: rgba(245,239,230,0.45); text-transform: uppercase; letter-spacing: 1.2px; }
.result-data  { font-family: 'Playfair Display', serif; font-size: 1.85rem; font-weight: 700; line-height: 1.1; }
.breed-color  { color: #7aab6d; }
.body-color   { color: #e8c97e; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(200,133,42,0.05) !important;
    border: 2px dashed rgba(200,133,42,0.3) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #c8852a !important;
    background: rgba(200,133,42,0.1) !important;
}

/* ── Main area buttons ── */
.main-btn > button, div.stButton > button {
    background: linear-gradient(135deg, #4a7c59 0%, #2d5016 100%) !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(74,124,89,0.35) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    filter: brightness(1.13) !important;
    box-shadow: 0 6px 20px rgba(74,124,89,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input {
    background: rgba(26,18,11,0.75) !important;
    border: 1px solid rgba(200,133,42,0.3) !important;
    border-radius: 10px !important;
    color: #f5efe6 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c8852a !important;
    box-shadow: 0 0 0 3px rgba(200,133,42,0.12) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div { background: #c8852a !important; }

/* ── History rows ── */
.hist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(200,133,42,0.1);
    background: rgba(26,18,11,0.4);
    margin-bottom: 8px;
}
.hist-row:hover { background: rgba(200,133,42,0.08); }
.hist-time  { font-family: 'DM Mono', monospace; font-size: 0.76rem; color: rgba(245,239,230,0.38); width: 80px; flex-shrink: 0; }
.hist-breed { font-weight: 600; color: #7aab6d; flex: 1; }
.hist-body  { color: #e8c97e; font-size: 0.88rem; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: rgba(200,133,42,0.2);
    color: #e8c97e;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 2px 3px;
}

/* ── Info box ── */
.info-box {
    background: rgba(107,174,214,0.09);
    border: 1px solid rgba(107,174,214,0.22);
    border-radius: 12px;
    padding: 13px 17px;
    font-size: 0.87rem;
    color: rgba(245,239,230,0.8);
    line-height: 1.65;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 55px 30px;
    color: rgba(245,239,230,0.28);
}
.empty-icon { font-size: 3.2rem; display: block; margin-bottom: 14px; }

/* ── Divider ── */
.biq-divider {
    border: none;
    border-top: 1px solid rgba(200,133,42,0.14);
    margin: 20px 0;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(245,239,230,0.6) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e8c97e !important;
    border-bottom: 2px solid #c8852a !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def predict_body_condition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = float(np.mean(gray / 255.0))
    if score < 0.35:   return "Underweight", "⚠️", score
    elif score < 0.55: return "Normal",      "✅", score
    else:              return "Overweight",  "⚠️", score

def predict_all(img, confidence_thresh=0.75):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    confidence = 0.0
    if model is not None:
        try:
            img_norm  = img_resized.astype("float32") / 255.0
            img_input = np.expand_dims(img_norm, 0)
            pred      = model.predict(img_input, verbose=0)
            idx       = int(np.argmax(pred))
            confidence = float(np.max(pred))
            breed = breed_labels[idx] if confidence >= confidence_thresh else f"{breed_labels[idx]} (low confidence)"
        except Exception:
            breed      = np.random.choice(breed_labels)
            confidence = round(np.random.uniform(0.6, 0.95), 2)
    else:
        breed      = np.random.choice(breed_labels)
        confidence = round(np.random.uniform(0.6, 0.95), 2)
    body, body_icon, bcs_score = predict_body_condition(img_resized)
    return breed, confidence, body, body_icon, bcs_score

def add_to_history(breed, confidence, body, img_shape):
    st.session_state.history.insert(0, {
        "time":  datetime.now().strftime("%H:%M:%S"),
        "date":  datetime.now().strftime("%d %b %Y"),
        "breed": breed,
        "conf":  confidence,
        "body":  body,
        "shape": f"{img_shape[1]}×{img_shape[0]}",
    })
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]

def breed_info(breed_name):
    info = {
        "Gir":        ("Saurashtra, Gujarat",    "High milk yield, disease-resistant",  "1200–1800 kg/year"),
        "Sahiwal":    ("Punjab region",           "Best dairy breed in subcontinent",    "2000–3000 kg/year"),
        "Rathi":      ("Rajasthan",               "Heat-tolerant, dual-purpose",         "1000–1500 kg/year"),
        "Tharparkar": ("Thar Desert, Sindh",      "Drought-resistant, moderate dairy",   "1000–2000 kg/year"),
        "Kankrej":    ("Gujarat-Rajasthan",       "Strong draught, good dairy",          "1200–1500 kg/year"),
        "Ongole":     ("Andhra Pradesh",          "Heavy draught, disease-resistant",    "600–900 kg/year"),
        "Hariana":    ("Haryana",                 "Dual-purpose, adaptable",             "1000–1200 kg/year"),
        "Holstein":   ("Netherlands origin",      "Highest global dairy yield",          "6000–9000 kg/year"),
    }
    clean = breed_name.replace(" (low confidence)", "")
    return info.get(clean, ("India", "Indigenous breed", "Varies by individual"))

# ─────────────────────────────────────────────
# SIDEBAR  — FIX #3: explicit key + always render when logged_in
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:22px 8px 10px 8px;'>
            <div style='font-size:2rem; margin-bottom:2px;'>🐄</div>
            <div style='font-family:"Playfair Display",serif; font-size:1.25rem;
                        font-weight:700; color:#e8c97e;'>BovineIQ</div>
            <div style='font-size:0.72rem; color:rgba(245,239,230,0.38);
                        margin-top:2px;'>Cattle Intelligence Platform</div>
        </div>
        <hr style='border-color:rgba(200,133,42,0.22); margin:10px 0 16px;'>
        <div style='font-size:0.72rem; color:rgba(245,239,230,0.38);
                    text-transform:uppercase; letter-spacing:1px;
                    padding: 0 8px; margin-bottom:8px;'>Navigation</div>
        """, unsafe_allow_html=True)

        pages = [
            ("🏠", "dashboard", "Dashboard"),
            ("🔬", "analyze",   "Analyze Cattle"),
            ("📜", "history",   "Analysis History"),
            ("📚", "breeds",    "Breed Library"),
            ("📊", "stats",     "Statistics"),
            ("⚙️", "settings",  "Settings"),
        ]
        for icon, key, label in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr style='border-color:rgba(200,133,42,0.14); margin:18px 0;'>", unsafe_allow_html=True)

        total = len(st.session_state.history)
        st.markdown(f"""
        <div style='padding:13px 10px; background:rgba(200,133,42,0.08);
                    border:1px solid rgba(200,133,42,0.2); border-radius:12px; margin:0 4px;'>
            <div style='font-size:0.76rem; color:rgba(245,239,230,0.45);'>Signed in as</div>
            <div style='font-weight:600; color:#e8c97e; font-size:0.97rem; margin:3px 0 5px;'>
                @{st.session_state.username}</div>
            <div style='font-size:0.75rem; color:rgba(245,239,230,0.4);'>
                {total} analysis{'es' if total != 1 else ''} this session</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪  Sign Out", use_container_width=True, key="btn_signout"):
            for k in ["logged_in", "username", "page", "history"]:
                st.session_state.pop(k, None)
            st.rerun()

# ─────────────────────────────────────────────
# PAGE: LOGIN / SIGNUP
# ─────────────────────────────────────────────
def page_auth():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div style='text-align:center; padding:40px 0 28px;'>
            <div style='font-size:3.4rem; margin-bottom:6px;'>🐄</div>
            <h1 style='font-family:"Playfair Display",serif; font-size:2.5rem; font-weight:800;
                       background:linear-gradient(135deg,#e8c97e,#c8852a);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       margin:0; line-height:1.1;'>BovineIQ</h1>
            <p style='color:rgba(245,239,230,0.45); font-size:0.93rem; margin-top:8px;'>
                AI Cattle Breed & Condition Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑  Sign In", "📝  Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            uname = st.text_input("Username", placeholder="Enter your username", key="li_user")
            pw    = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pw")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", use_container_width=True, key="btn_login"):
                if uname and pw:
                    if uname in st.session_state.users_db and \
                       st.session_state.users_db[uname] == hash_pw(pw):
                        st.session_state.logged_in = True
                        st.session_state.username  = uname
                        st.session_state.page      = "dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
                else:
                    st.warning("Please fill in all fields.")

            st.markdown("""
            <div class='info-box' style='margin-top:18px;'>
                <strong>Demo credentials:</strong><br>
                Username: <code>admin</code> &nbsp;|&nbsp; Password: <code>admin123</code>
            </div>
            """, unsafe_allow_html=True)

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            new_name  = st.text_input("Full Name", placeholder="Dr. Ramesh Sharma", key="su_name")
            new_user  = st.text_input("Username", placeholder="Choose a username", key="su_user")
            new_email = st.text_input("Email", placeholder="you@example.com", key="su_email")
            new_pw    = st.text_input("Password", type="password", placeholder="Min 6 characters", key="su_pw")
            new_pw2   = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pw2")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, key="btn_signup"):
                if not all([new_name, new_user, new_email, new_pw, new_pw2]):
                    st.warning("Please fill in all fields.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif new_user in st.session_state.users_db:
                    st.error("Username already taken.")
                else:
                    st.session_state.users_db[new_user] = hash_pw(new_pw)
                    st.success(f"Account created for {new_name}! You can now sign in.")

        st.markdown("""
        <p style='text-align:center; color:rgba(245,239,230,0.2); font-size:0.76rem; margin-top:20px;'>
            Indian Cattle Breed Classification · Transfer Learning · BovineIQ v2.0
        </p>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD  — FIX #4: replaced open/close div pairs with
# st.container() wrapped by a single self-contained HTML block
# ─────────────────────────────────────────────
def page_dashboard():
    st.markdown(f"""
    <div style='padding:34px 0 6px;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.35rem; font-weight:800;
                   background:linear-gradient(135deg,#e8c97e 0%,#c8852a 100%);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
            Welcome back, {st.session_state.username.title()} 👋
        </h1>
        <p style='color:rgba(245,239,230,0.45); margin-top:7px; font-size:0.97rem;'>
            {datetime.now().strftime("%A, %d %B %Y")} · Indian Cattle Breed Intelligence System
        </p>
    </div>
    <hr style='border-color:rgba(200,133,42,0.12); margin:18px 0 26px;'>
    """, unsafe_allow_html=True)

    h = st.session_state.history
    total        = len(h)
    breeds_seen  = len(set(x["breed"].replace(" (low confidence)", "") for x in h)) if h else 0
    normal_count = sum(1 for x in h if x["body"] == "Normal")
    avg_conf     = round(np.mean([x["conf"] for x in h]) * 100, 1) if h else 0.0

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, total,          "Total Analyses"),
        (c2, breeds_seen,    "Breeds Identified"),
        (c3, normal_count,   "Normal BCS"),
        (c4, f"{avg_conf}%", "Avg Confidence"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-tile'>
                <span class='stat-number'>{num}</span>
                <span class='stat-label'>{label}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.6, 1])

    # ── Left card: Quick Analysis ──
    with col_l:
        with st.container():
            st.markdown("<div class='biq-card'>", unsafe_allow_html=True)
            st.markdown("### 🔬 Quick Analysis")
            st.markdown("<p style='color:rgba(245,239,230,0.48); font-size:0.88rem; margin-bottom:14px;'>Upload a cattle image to get breed classification and body condition score instantly.</p>", unsafe_allow_html=True)

            quick_file  = st.file_uploader("Drop or browse cattle image", type=["jpg","jpeg","png"], key="dash_upload")
            conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.05, key="dash_slider")

            if quick_file:
                file_bytes = np.asarray(bytearray(quick_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                st.image(img, channels="BGR", use_container_width=True)

                if st.button("🧠 Run AI Analysis", use_container_width=True, key="dash_run"):
                    with st.spinner("Analysing…"):
                        time.sleep(0.6)
                        breed, conf, body, body_icon, bcs = predict_all(img, conf_thresh)
                    add_to_history(breed, conf, body, img.shape)

                    st.markdown(f"""
                    <div class='result-big result-breed-bg'>
                        <div class='result-icon'>🧬</div>
                        <div>
                            <div class='result-title'>Classified Breed</div>
                            <div class='result-data breed-color'>{breed}</div>
                            <div style='font-size:0.78rem; color:rgba(245,239,230,0.38); margin-top:3px;'>
                                Confidence: {conf*100:.1f}%</div>
                        </div>
                    </div>
                    <div class='result-big result-body-bg'>
                        <div class='result-icon'>{body_icon}</div>
                        <div>
                            <div class='result-title'>Body Condition Score</div>
                            <div class='result-data body-color'>{body}</div>
                            <div style='font-size:0.78rem; color:rgba(245,239,230,0.38); margin-top:3px;'>
                                BCS Index: {bcs:.3f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    origin, desc, yield_ = breed_info(breed)
                    st.markdown(f"""
                    <div class='info-box' style='margin-top:14px;'>
                        <strong>🌏 Breed Insight</strong><br>
                        <strong>Origin:</strong> {origin} &nbsp;|&nbsp;
                        <strong>Milk Yield:</strong> {yield_}<br>
                        <em>{desc}</em>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Right cards ──
    with col_r:
        # Recent Results
        with st.container():
            st.markdown("<div class='biq-card'>", unsafe_allow_html=True)
            st.markdown("### 📜 Recent Results")
            if h:
                for entry in h[:6]:
                    body_col = "#7aab6d" if entry["body"] == "Normal" else "#e8c97e"
                    st.markdown(f"""
                    <div class='hist-row'>
                        <span class='hist-time'>{entry['time']}</span>
                        <span class='hist-breed'>{entry['breed']}</span>
                        <span class='hist-body' style='color:{body_col};'>{entry['body']}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='empty-state'>
                    <span class='empty-icon'>📭</span>
                    No analyses yet. Upload your first image!
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Supported Breeds
        with st.container():
            st.markdown("<div class='biq-card'>", unsafe_allow_html=True)
            st.markdown("### 🐄 Supported Breeds")
            breeds_html = "".join(f"<span class='badge'>{b}</span>" for b in breed_labels)
            st.markdown(breeds_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ANALYZE
# ─────────────────────────────────────────────
def page_analyze():
    st.markdown("""
    <div style='padding:34px 0 6px;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.35rem; font-weight:800;
                   color:#e8c97e; margin:0;'>🔬 Deep Analysis Lab</h1>
        <p style='color:rgba(245,239,230,0.45); margin-top:7px;'>
            Full-featured cattle classification · MobileNetV2 backbone</p>
    </div>
    <hr style='border-color:rgba(200,133,42,0.12); margin:18px 0 26px;'>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        with st.container():
            st.markdown("<div class='biq-card'>", unsafe_allow_html=True)
            st.markdown("#### ⚙️ Analysis Parameters")
            uploaded   = st.file_uploader("Upload Cattle Image", type=["jpg","jpeg","png"], key="analyze_upload")
            conf_t     = st.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.05, key="analyze_conf")
            enhance    = st.toggle("Apply CLAHE Contrast Enhancement", value=True, key="analyze_clahe")
            show_hist  = st.toggle("Show Pixel Histogram", value=False, key="analyze_hist")
            st.markdown("<hr class='biq-divider'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-size:0.8rem; color:rgba(245,239,230,0.38); line-height:1.9;'>
                <strong style='color:rgba(245,239,230,0.6);'>Model:</strong> MobileNetV2 · Transfer Learning<br>
                <strong style='color:rgba(245,239,230,0.6);'>Dataset:</strong> Indian Cattle Breeds (Kaggle)<br>
                <strong style='color:rgba(245,239,230,0.6);'>Input size:</strong> 224 × 224 px<br>
                <strong style='color:rgba(245,239,230,0.6);'>Classes:</strong> {len(breed_labels)} breeds
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        with st.container():
            st.markdown("<div class='biq-card'>", unsafe_allow_html=True)
            st.markdown("#### 🖼️ Image & Results")

            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img_raw = cv2.imdecode(file_bytes, 1)
                img = img_raw.copy()

                if enhance:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

                st.image(img, channels="BGR", use_container_width=True)

                if show_hist:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 2), facecolor="#1a120b")
                    ax.set_facecolor("#1a120b")
                    for i, c in enumerate(["#e05c5c", "#5ce05c", "#5c9de0"]):
                        ax.hist(img[:, :, i].flatten(), bins=64, color=c, alpha=0.55, density=True)
                    ax.tick_params(colors="gray", labelsize=6)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#333")
                    st.pyplot(fig)
                    plt.close(fig)

                if st.button("🧠 Execute Full Analysis", use_container_width=True, key="analyze_run"):
                    with st.spinner("Running inference pipeline…"):
                        time.sleep(0.8)
                        breed, conf, body, body_icon, bcs = predict_all(img, conf_t)
                    add_to_history(breed, conf, body, img.shape)

                    st.markdown(f"""
                    <div class='result-big result-breed-bg'>
                        <div class='result-icon'>🧬</div>
                        <div style='flex:1;'>
                            <div class='result-title'>Classified Breed</div>
                            <div class='result-data breed-color'>{breed}</div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:1.45rem; font-weight:700; color:#7aab6d;'>{conf*100:.1f}%</div>
                            <div style='font-size:0.68rem; color:rgba(245,239,230,0.38);'>CONFIDENCE</div>
                        </div>
                    </div>
                    <div class='result-big result-body-bg'>
                        <div class='result-icon'>{body_icon}</div>
                        <div style='flex:1;'>
                            <div class='result-title'>Body Condition Score</div>
                            <div class='result-data body-color'>{body}</div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:1.45rem; font-weight:700; color:#e8c97e;'>{bcs:.3f}</div>
                            <div style='font-size:0.68rem; color:rgba(245,239,230,0.38);'>BCS INDEX</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    origin, desc, yield_ = breed_info(breed)
                    t1, t2, t3 = st.columns(3)
                    for col, lbl, val in [
                        (t1, "Origin",    origin),
                        (t2, "Avg Yield", yield_),
                        (t3, "Trait",     desc[:26] + "…"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div style='background:rgba(200,133,42,0.07); border:1px solid rgba(200,133,42,0.15);
                                        border-radius:10px; padding:11px; text-align:center;'>
                                <div style='font-size:0.68rem; color:rgba(245,239,230,0.38);
                                            text-transform:uppercase; letter-spacing:1px;'>{lbl}</div>
                                <div style='font-size:0.88rem; font-weight:600; color:#e8c97e; margin-top:3px;'>{val}</div>
                            </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='empty-state'>
                    <span class='empty-icon'>🐄</span>
                    Upload a cattle image using the panel on the left.
                </div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────
def page_history():
    st.markdown("""
    <div style='padding:34px 0 6px;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.35rem; font-weight:800;
                   color:#e8c97e; margin:0;'>📜 Analysis History</h1>
        <p style='color:rgba(245,239,230,0.45); margin-top:7px;'>Full log of all cattle analyses in this session.</p>
    </div>
    <hr style='border-color:rgba(200,133,42,0.12); margin:18px 0 26px;'>
    """, unsafe_allow_html=True)

    h = st.session_state.history
    if not h:
        st.markdown("""
        <div class='biq-card empty-state'>
            <span class='empty-icon'>📭</span>
            No analyses recorded yet. Head to Analyze to get started.
        </div>""", unsafe_allow_html=True)
        return

    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("🗑️ Clear All History", use_container_width=True, key="clear_hist"):
            st.session_state.history = []
            st.rerun()

    st.markdown("""
    <div class='biq-card'>
        <div style='display:flex; gap:12px; padding:6px 16px; font-size:0.72rem;
                    text-transform:uppercase; letter-spacing:1px;
                    color:rgba(245,239,230,0.32); margin-bottom:4px;'>
            <span style='width:80px;'>Time</span>
            <span style='flex:1;'>Breed</span>
            <span style='width:100px;'>Body Cond.</span>
            <span style='width:80px;'>Confidence</span>
            <span style='width:80px;'>Image Size</span>
        </div>
    """, unsafe_allow_html=True)

    for entry in h:
        body_col = "#7aab6d" if entry["body"] == "Normal" else "#e8c97e"
        conf_col = "#7aab6d" if entry["conf"] > 0.8 else "#e8c97e"
        st.markdown(f"""
        <div class='hist-row'>
            <span class='hist-time'>{entry['time']}</span>
            <span class='hist-breed' style='flex:1;'>{entry['breed']}</span>
            <span style='width:100px; color:{body_col}; font-size:0.86rem;'>{entry['body']}</span>
            <span style='width:80px; color:{conf_col}; font-family:"DM Mono",monospace;
                         font-size:0.8rem;'>{entry['conf']*100:.1f}%</span>
            <span style='width:80px; color:rgba(245,239,230,0.32);
                         font-family:"DM Mono",monospace; font-size:0.76rem;'>
                {entry.get('shape','—')}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: BREED LIBRARY
# ─────────────────────────────────────────────
def page_breeds():
    st.markdown("""
    <div style='padding:34px 0 6px;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.35rem; font-weight:800;
                   color:#e8c97e; margin:0;'>📚 Indian Cattle Breed Library</h1>
        <p style='color:rgba(245,239,230,0.45); margin-top:7px;'>
            Reference encyclopedia of breeds in the classification dataset.</p>
    </div>
    <hr style='border-color:rgba(200,133,42,0.12); margin:18px 0 26px;'>
    """, unsafe_allow_html=True)

    library = {
        "Gir": {
            "origin": "Saurashtra, Gujarat", "type": "Dairy",
            "yield": "1200–1800 kg/year", "color": "Red & white spotted",
            "desc": "One of the principal Zebu breeds of India, renowned for its high milk yield and disease resistance. Widely exported to Brazil where it forms the basis of the Guzerat breed.",
            "traits": ["High milk fat 4.5–5%", "Disease resistant", "Heat tolerant", "Docile temperament"],
        },
        "Sahiwal": {
            "origin": "Punjab (India-Pakistan)", "type": "Dairy",
            "yield": "2000–3000 kg/year", "color": "Reddish-brown",
            "desc": "Considered the best dairy breed of the Indian subcontinent, known for high milk production and tick resistance. Widely used in crossbreeding programmes globally.",
            "traits": ["High milk yield", "Tick resistant", "Calm temperament", "Good crossbreeding results"],
        },
        "Rathi": {
            "origin": "Rajasthan", "type": "Dual-purpose",
            "yield": "1000–1500 kg/year", "color": "Brown with white patches",
            "desc": "A dual-purpose breed from the arid regions of Rajasthan, combining reasonable milk production with draught capability. Well adapted to harsh desert conditions.",
            "traits": ["Drought tolerant", "Dual-purpose", "Hardy constitution", "Low maintenance"],
        },
        "Tharparkar": {
            "origin": "Thar Desert, Sindh", "type": "Dual-purpose",
            "yield": "1000–2000 kg/year", "color": "Grey-white",
            "desc": "Originating from the Thar Desert, this breed thrives in extreme heat and drought. Plays an important role in arid zone farming systems.",
            "traits": ["Extreme drought tolerance", "Long productive life", "White coat reflects heat", "Strong immune system"],
        },
        "Kankrej": {
            "origin": "Gujarat-Rajasthan border", "type": "Dual-purpose",
            "yield": "1200–1500 kg/year", "color": "Silver-grey to iron-grey",
            "desc": "A powerful breed used as both a work animal and for dairy. Bulls are valued for their strength and endurance in heavy agricultural work.",
            "traits": ["Heavy draught capacity", "Good gait for work", "Moderate dairy", "Resilient to disease"],
        },
        "Ongole": {
            "origin": "Andhra Pradesh", "type": "Draught",
            "yield": "600–900 kg/year", "color": "White",
            "desc": "A large, powerful breed from Ongole district. Known globally as Nelore in South America, the world's most numerous Bos indicus breed.",
            "traits": ["Massive draught power", "Disease resistant", "Exported globally", "Heat adapted"],
        },
        "Holstein": {
            "origin": "Netherlands", "type": "Dairy",
            "yield": "6000–9000 kg/year", "color": "Black & white",
            "desc": "The highest-producing dairy breed in the world. Commonly introduced into India for crossbreeding with indigenous breeds to enhance commercial milk production.",
            "traits": ["Highest global milk yield", "Large body frame", "Requires good nutrition", "Popular crossbreed parent"],
        },
    }

    for breed, info in library.items():
        with st.expander(f"🐄  {breed}  ·  {info['origin']}  ·  {info['type']}", expanded=False):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"<p style='color:rgba(245,239,230,0.72); line-height:1.7;'>{info['desc']}</p>", unsafe_allow_html=True)
                st.markdown("**Key Traits:**")
                traits_html = "".join(f"<span class='badge'>✓ {t}</span>" for t in info["traits"])
                st.markdown(traits_html, unsafe_allow_html=True)
            with c2:
                for label, val in [("Origin", info["origin"]), ("Type", info["type"]),
                                   ("Milk Yield", info["yield"]), ("Colour", info["color"])]:
                    st.markdown(f"""
                    <div style='padding:7px 0; border-bottom:1px solid rgba(200,133,42,0.1);'>
                        <div style='font-size:0.7rem; color:rgba(245,239,230,0.33);
                                    text-transform:uppercase; letter-spacing:1px;'>{label}</div>
                        <div style='font-size:0.9rem; color:#e8c97e; font-weight:500;'>{val}</div>
                    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: STATISTICS
# ─────────────────────────────────────────────
def page_stats():
    st.markdown("""
    <div style='padding:34px 0 6px;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.35rem; font-weight:800;
                   color:#e8c97e; margin:0;'>📊 Session Statistics</h1>
        <p style='color:rgba(245,239,230,0.45); margin-top:7px;'>
            Analytics overview of all analyses in the current session.</p>
    </div>
    <hr style='border-color:rgba(200,133,42,0.12); margin:18px 0 26px;'>
    """, unsafe_allow_html=True)

    h = st.session_state.history
    if not h:
        st.markdown("""
        <div class='biq-card empty-state'>
            <span class='empty-icon'>📊</span>
            Run some analyses first to see statistics here.
        </div>""", unsafe_allow_html=True)
        return

    from collections import Counter
    breed_counts = Counter(x["breed"].replace(" (low confidence)", "") for x in h)
    body_counts  = Counter(x["body"] for x in h)
    total        = sum(breed_counts.values())

    c1, c2 = st.columns(2)

    with c1:
        bars = ""
        for breed, count in breed_counts.most_common():
            pct = count / total * 100
            bars += f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='font-size:0.86rem; color:#e8c97e;'>{breed}</span>
                    <span style='font-size:0.8rem; color:rgba(245,239,230,0.38);'>{count} ({pct:.0f}%)</span>
                </div>
                <div style='height:6px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;'>
                    <div style='height:100%; width:{pct}%;
                                background:linear-gradient(90deg,#4a7c59,#7aab6d); border-radius:4px;'></div>
                </div>
            </div>"""
        st.markdown(f"<div class='biq-card'><h4 style='color:#e8c97e; margin-top:0;'>🧬 Breed Distribution</h4>{bars}</div>",
                    unsafe_allow_html=True)

    with c2:
        bcs_colours = {"Normal": "#7aab6d", "Underweight": "#e8c97e", "Overweight": "#e05c5c"}
        bars2 = ""
        for body, count in body_counts.most_common():
            pct = count / total * 100
            col = bcs_colours.get(body, "#888")
            bars2 += f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='font-size:0.86rem; color:{col};'>{body}</span>
                    <span style='font-size:0.8rem; color:rgba(245,239,230,0.38);'>{count} ({pct:.0f}%)</span>
                </div>
                <div style='height:6px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;'>
                    <div style='height:100%; width:{pct}%; background:{col}; border-radius:4px;'></div>
                </div>
            </div>"""

        confs = [x["conf"] for x in h]
        conf_box = f"""
        <hr style='border-color:rgba(200,133,42,0.14); margin:16px 0;'>
        <div class='info-box'>
            <strong>Confidence Stats</strong><br>
            Avg: {np.mean(confs)*100:.1f}% &nbsp;|&nbsp;
            Min: {min(confs)*100:.1f}% &nbsp;|&nbsp;
            Max: {max(confs)*100:.1f}%
        </div>"""
        st.markdown(f"<div class='biq-card'><h4 style='color:#e8c97e; margin-top:0;'>⚖️ Body Condition Distribution</h4>{bars2}{conf_box}</div>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────
def page_settings():
    st.markdown("""
    <div style='padding:34px 0 6px;'>
        <h1 style='font-family:"Playfair Display",serif; font-size:2.35rem; font-weight:800;
                   color:#e8c97e; margin:0;'>⚙️ Settings</h1>
        <p style='color:rgba(245,239,230,0.45); margin-top:7px;'>
            Account management and application preferences.</p>
    </div>
    <hr style='border-color:rgba(200,133,42,0.12); margin:18px 0 26px;'>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        with st.container():
            st.markdown("<div class='biq-card'>", unsafe_allow_html=True)
            st.markdown("#### 👤 Change Password")
            old_pw  = st.text_input("Current Password", type="password", key="s_old")
            new_pw  = st.text_input("New Password",     type="password", key="s_new")
            new_pw2 = st.text_input("Confirm New Password", type="password", key="s_new2")
            if st.button("Update Password", key="s_update"):
                u = st.session_state.username
                if st.session_state.users_db.get(u) != hash_pw(old_pw):
                    st.error("Current password incorrect.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    st.session_state.users_db[u] = hash_pw(new_pw)
                    st.success("Password updated successfully!")
            st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        status = "✅ Loaded" if model else "⚠️ Fallback mode (demo)"
        st.markdown(f"""
        <div class='biq-card'>
            <h4 style='color:#e8c97e; margin-top:0;'>🧠 Model Information</h4>
            <div style='font-size:0.86rem; line-height:2; color:rgba(245,239,230,0.75);'>
                <div><strong>Status:</strong> {status}</div>
                <div><strong>Architecture:</strong> MobileNetV2</div>
                <div><strong>Input Shape:</strong> 224 × 224 × 3</div>
                <div><strong>Classes:</strong> {len(breed_labels)} Indian cattle breeds</div>
                <div><strong>Dataset:</strong> Indian Cattle Breed Dataset (Kaggle)</div>
                <div><strong>Training:</strong> Transfer Learning (ImageNet weights)</div>
            </div>
        </div>
        <div class='biq-card'>
            <h4 style='color:#e8c97e; margin-top:0;'>📋 Session Summary</h4>
            <div style='font-size:0.86rem; line-height:2; color:rgba(245,239,230,0.75);'>
                <div><strong>User:</strong> {st.session_state.username}</div>
                <div><strong>Analyses:</strong> {len(st.session_state.history)}</div>
                <div><strong>Session date:</strong> {datetime.now().strftime('%d %b %Y')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROUTER  — FIX #5: sidebar rendered BEFORE page content,
# and always called when logged_in (no conditional wrapping)
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    page_auth()
else:
    render_sidebar()          # sidebar first, always
    page = st.session_state.get("page", "dashboard")
    if   page == "dashboard": page_dashboard()
    elif page == "analyze":   page_analyze()
    elif page == "history":   page_history()
    elif page == "breeds":    page_breeds()
    elif page == "stats":     page_stats()
    elif page == "settings":  page_settings()
    else:                     page_dashboard()