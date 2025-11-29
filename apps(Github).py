import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import pickle
import time
import ollama
import os
from groq import Groq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="UrbanPulse: GovDSS",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL UI STYLING (CSS) ---
st.markdown("""
<style>
    /* MAIN THEME */
    .stApp { background-color: #0E1117; font-family: 'Segoe UI', Roboto, sans-serif; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    
    /* METRIC CARDS */
    div[data-testid="metric-container"] {
        background-color: #161B22; border: 1px solid #30363D; padding: 20px;
        border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-2px); border-color: #58A6FF; }
    
    /* BUTTONS */
    div.stButton > button {
        width: 100%; height: 60px; border-radius: 8px; border: 1px solid #30363D;
        background-color: #21262D; color: #C9D1D9; font-weight: 600; transition: all 0.2s;
    }
    div.stButton > button:hover { background-color: #30363D; color: #58A6FF; border-color: #58A6FF; }
    div.stButton > button:active { background-color: #1F6FEB; color: white; }

    /* STATUS BADGE */
    .status-live {
        background-color: #238636; color: white; padding: 4px 10px;
        border-radius: 20px; font-size: 12px; font-weight: bold; vertical-align: middle; margin-left: 10px;
    }
    
    /* AI BOX */
    .ai-box {
        background-color: #1F2937; border-left: 4px solid #8B5CF6; padding: 15px;
        border-radius: 5px; color: #E5E7EB; font-size: 14px; line-height: 1.5;
    }
    
    /* HERO BANNER GRADIENT OVERLAY */
    .banner-container {
        position: relative;
        text-align: center;
        color: white;
    }
    .banner-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(14,17,23,1) 100%);
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA & CONFIG ---
CLIMATE_DATA = {"Annual_Rainy_Days": 210, "Rain_Probability": 0.57}

LOCATIONS = {
    "Jalan Tun Razak (Commercial)": {
        "id": "Tun_Razak", "coords": [3.1579, 101.7116], "district": "Bukit Bintang", 
        "density": 11000, "base_traffic": 8500, "vulnerable_pop": 0.12
    },
    "Bangsar South (Mixed Usage)": {
        "id": "Bangsar", "coords": [3.1110, 101.6650], "district": "Bangsar", 
        "density": 6500, "base_traffic": 6000, "vulnerable_pop": 0.08
    },
    "Cheras Utama (Residential)": {
        "id": "Cheras", "coords": [3.0550, 101.7560], "district": "Cheras", 
        "density": 9500, "base_traffic": 7000, "vulnerable_pop": 0.15
    }
}

# --- 4. ENGINES ---
class UrbanPulseAI:
    def __init__(self):
        try:
            with open("model_stress.pkl", "rb") as f: self.stress_model = pickle.load(f)
            with open("model_roi.pkl", "rb") as f: self.roi_model = pickle.load(f)
            with open("encoder.pkl", "rb") as f: self.le = pickle.load(f)
            self.loaded = True
        except: self.loaded = False

    def predict(self, loc_data, intervention_code):
        if not self.loaded: return 0,0,0, {}
        i_code_map = {"Trees": 1, "Bike": 2, "Emergency": 3, "Flyover": 4, "PublicTransport": 5}
        i_code = i_code_map.get(intervention_code, 0)
        
        # Weighted Climate Logic (Government Grade)
        t_sunny = loc_data["base_traffic"] + 500
        if i_code == 2: t_sunny *= 0.85
        if i_code == 5: t_sunny *= 0.70
        t_rain = loc_data["base_traffic"]
        if i_code == 2: t_rain *= 0.98
        if i_code == 5: t_rain *= 0.75
        avg_traffic = (t_sunny * 0.43) + (t_rain * 0.57)
        if i_code == 4: avg_traffic *= 1.15 # Induced demand

        # ML Prediction
        ml_i_code = i_code if i_code <= 3 else 0 
        try: dist_code = self.le.transform([loc_data["district"]])[0]
        except: dist_code = 0
        
        inputs = pd.DataFrame([[avg_traffic, 1, ml_i_code, loc_data["density"], dist_code]], 
                              columns=["traffic", "weather", "intervention", "density", "district_code"])
        stress = self.stress_model.predict(inputs)[0]
        roi = self.roi_model.predict(inputs)[0]
        
        # Manual Adjustments
        if i_code == 4: stress += 15; roi -= 1.5
        if i_code == 5: stress -= 10; roi += 4.5
        
        breakdown = {"Healthcare (Asthma)": roi * 0.6, "Productivity": roi * 0.3, "Fuel Savings": roi * 0.1}
        return round(stress, 1), round(roi, 2), int(avg_traffic), breakdown

class SeaLionBrain:
    def __init__(self):
        # ⚠️ PASTE GROQ KEY HERE
        self.api_key = "PASTE_YOUR_GROQ_API_KEY_HERE" 
        try:
            self.client = Groq(api_key=self.api_key)
            self.online = True
        except: self.online = False

    def ask_copilot(self, location, intervention, stress, roi, climate_txt):
        if not self.online: return "⚠️ API Key Missing."
        
        system_prompt = """
        ROLE: You are 'SEA-LION', an AI Town Planner for DBKL (Kuala Lumpur).
        TONE: Formal Malaysian Government style (Bahasa Baku + Professional English).
        INSTRUCTIONS: 
        1. Review the data. 
        2. If 'Flyover' selected, warn about Induced Demand. 
        3. If 'Bike Lane' selected, warn about Monsoon rain impact.
        4. Keep under 3 sentences.
        """
        user_prompt = f"Project: {location}+{intervention}. Stress:{stress}, ROI:{roi}M. Context:{climate_txt}"
        
        try:
            return self.client.chat.completions.create(
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            ).choices[0].message.content
        except Exception as e: return f"API Error: {str(e)}"

math_engine = UrbanPulseAI()
ai_brain = SeaLionBrain()

# --- 5. LAYOUT & LOGIC ---
if 'active_tool' not in st.session_state: st.session_state.active_tool = "None"

# HEADER WITH W.A.R.D BANNER
if os.path.exists("banner.jpg"):
    st.image("banner.jpeg", use_container_width=True)
else:
    # Fallback header if image missing
    st.title("W.A.R.D: Urban Intelligence")

st.markdown("""
    <div style='display: flex; align-items: center; margin-top: 10px;'>
        <h1 style='margin: 0; padding: 0; font-size: 2.5rem;'>W.A.R.D <span style='color: #58A6FF; font-size: 1.5rem;'>GovDSS</span></h1>
        <span class='status-live'>● SYSTEM ONLINE</span>
    </div>
    <p style='margin: 0; opacity: 0.7; font-size: 1.1rem;'>Weighted Analytics for Resilient Development (KL Division)</p>
    <hr style='border-color: #30363D;'>
""", unsafe_allow_html=True)

# MAIN WORKSPACE
col_sidebar, col_main = st.columns([1, 3])

# --- LEFT PANEL (CONTROLS) ---
with col_sidebar:
    st.markdown("### 📍 Project Context")
    selected_loc_name = st.selectbox("Select District", list(LOCATIONS.keys()), label_visibility="collapsed")
    loc_data = LOCATIONS[selected_loc_name]
    
    # District Stats Card
    st.info(f"""
    **District:** {loc_data['district']}
    **Density:** {loc_data['density']:,} /km²
    **Base Traffic:** {loc_data['base_traffic']:,} pcu/hr
    """)
    
    st.markdown("### 🛠️ Intervention Dock")
    
    # 2x3 Grid Buttons
    r1c1, r1c2 = st.columns(2)
    with r1c1: 
        if st.button("🚫 Clear"): st.session_state.active_tool = "None"
        if st.button("🚴 Bike Lane"): st.session_state.active_tool = "Bike"
        if st.button("🛣️ Flyover"): st.session_state.active_tool = "Flyover"
    with r1c2:
        if st.button("🌳 Green Way"): st.session_state.active_tool = "Trees"
        if st.button("🏥 EMS Route"): st.session_state.active_tool = "Emergency"
        if st.button("🚌 Transit"): st.session_state.active_tool = "PublicTransport"
        
    st.markdown("---")
    st.caption("Data Sources: MetMalaysia, OpenDOSM, MOH")

# --- RIGHT PANEL (VISUALIZATION) ---
with col_main:
    curr = st.session_state.active_tool
    
    # Run Simulation
    if curr == "None":
        s, r, t, bd = 85.0, 0.0, loc_data["base_traffic"], {}
        ai_msg = "Ready for simulation inputs."
    else:
        if 'last_run' not in st.session_state or st.session_state.last_run != curr:
            st.toast(f"Simulating Impact: {curr}...", icon="🔄")
            st.session_state.last_run = curr
            
        s, r, t, bd = math_engine.predict(loc_data, curr)
        # Pass the "Weighted Average" context string to AI
        ai_msg = ai_brain.ask_copilot(selected_loc_name, curr, s, r, "Tropical (57% Rain Probability)") if math_engine.loaded else "Models missing."

    # KPIS ROW (Styled Cards)
    k1, k2, k3 = st.columns(3)
    k1.metric("Community Stress", f"{s}/100", delta="-Score (Better)" if s<85 else "+Score (Worse)", delta_color="inverse")
    k2.metric("Traffic Volume", f"{t} /hr", delta="Annual Avg")
    k3.metric("Health ROI", f"RM {r} M", delta="Annual Savings")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # MAP & INTELLIGENCE SPLIT
    tab_map, tab_detail = st.tabs(["🗺️ Digital Twin Map", "📊 Deep Dive Analysis"])
    
    with tab_map:
        mc1, mc2 = st.columns([2.5, 1])
        with mc1:
            lat, lon = loc_data["coords"]
            color = [255, 50, 50, 180] if s > 70 else [0, 200, 100, 180]
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14.5, pitch=50),
                layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame({'lat':[lat],'lon':[lon]}), get_position='[lon,lat]', get_color=color, get_radius=500, pickable=True)],
                map_style=pdk.map_styles.CARTO_DARK
            ), key=selected_loc_name)
        
        with mc2:
            st.markdown("#### 🤖 AI Policy Audit")
            st.markdown(f"<div class='ai-box'><b>Analisis SEA-LION:</b><br>{ai_msg}</div>", unsafe_allow_html=True)
            if curr == "Flyover":
                st.error("⚠️ **Warning:** Induced Demand Detected.")
    
    with tab_detail:
        c_dem, c_fin = st.columns(2)
        with c_dem:
            st.markdown("#### 👥 Demographic Impact")
            affected = int(loc_data["density"] * 1.5)
            vuln = int(affected * loc_data["vulnerable_pop"])
            st.bar_chart(pd.DataFrame({"Group": ["Adults", "Vulnerable (Kids/Elderly)"], "Count": [affected-vuln, vuln]}).set_index("Group"), color="#58A6FF")
        
        with c_fin:
            st.markdown("#### 💰 ROI Ledger")
            if r > 0:
                df_fin = pd.DataFrame(list(bd.items()), columns=["Category", "RM Millions"])
                st.dataframe(df_fin, hide_index=True, use_container_width=True)
            else:
                st.warning("Project ROI is negative/neutral.")
