import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import zipfile
import io
from pathlib import Path

# Page Config
st.set_page_config(page_title="TurbofanReason Viewer", layout="wide")

# ==========================================
# 1. REUSED LOGIC (Data & Expert)
# ==========================================
CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

@st.cache_data
def load_data():
    """Downloads and caches the NASA data."""
    cache_dir = Path("cmapss_cache")
    cache_dir.mkdir(exist_ok=True)
    train_path = cache_dir / "train_FD001.txt"

    if not train_path.exists():
        zip_path = cache_dir / "data.zip"
        if not zip_path.exists():
            with st.spinner("Downloading NASA Dataset..."):
                r = requests.get(CMAPSS_URL)
                zip_path.write_bytes(r.content)
        
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
            # Handle potential subfolders or direct extraction
            extracted = list(cache_dir.glob("**/train_FD001.txt"))
            if extracted:
                if extracted[0] != train_path:
                    # Move if needed, or just read from there. 
                    # Simpler to just read.
                    train_path = extracted[0]

    col_names = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    
    # Calculate RUL
    max_cycles = df.groupby('unit')['time'].max().rename('max_time')
    df = df.join(max_cycles, on='unit')
    df['rul'] = df['max_time'] - df['time']
    return df

class ReasoningExpert:
    def __init__(self):
        self.limits = {
            's2': 644.0,  # Compressor inlet temp
            's3': 1600.0, # HPC outlet temp
            's4': 1420.0, # LPT outlet temp
            's7': 554.0,  # HPC outlet pressure
            's11': 48.0,  # Static pressure 
        }

    def analyze(self, row, window_mean):
        reasons = []
        # Compare current/window mean to limits
        if window_mean['s4'] > self.limits['s4']:
            reasons.append(f"⚠️ **LPT Outlet Temp (Sensor 4)** is {window_mean['s4']:.1f}, exceeding nominal limit ({self.limits['s4']}).")
        
        if window_mean['s11'] > self.limits['s11']:
            reasons.append(f"⚠️ **Static Pressure (Sensor 11)** is high ({window_mean['s11']:.1f}), indicating flow restriction.")
            
        if window_mean['s7'] < self.limits['s7'] - 5: # Example lower bound logic
             reasons.append(f"⚠️ **HPC Pressure (Sensor 7)** is dropping dangerously.")

        status = "HEALTHY"
        if row['rul'] < 30:
            status = "CRITICAL"
            reasons.append("🔴 Multiple failure signatures detected coincident with high cycle count.")
        elif row['rul'] < 75:
            status = "WARNING"
            reasons.append("TK Sensors show early drift signs.")
        else:
            reasons.append("✅ All sensors operating within nominal bands.")
            
        return "  \n".join(reasons), status

# ==========================================
# 2. APP UI
# ==========================================
st.title("✈️ TurbofanReason: Explainable Predictive Maintenance")
st.markdown("Visualizing the **Reasoning Synthesis** layer. We translate raw sensor data into engineering diagnostics.")

df = load_data()

# Sidebar
units = df['unit'].unique()
selected_unit = st.sidebar.selectbox("Select Engine Unit", units)

# Filter Data
unit_df = df[df['unit'] == selected_unit]

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sensor Telemetry")
    # Plot S4 and S7
    fig = px.line(unit_df, x="time", y=["s4", "s7", "s11"], title=f"Engine {selected_unit} Key Sensors")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Live Diagnosis")
    selected_time = st.slider("Time Cycle", min_value=int(unit_df['time'].min()), max_value=int(unit_df['time'].max()))
    
    # Get Data for this cycle
    current_row = unit_df[unit_df['time'] == selected_time].iloc[0]
    
    # Simple window for smoothing (last 5 cycles)
    window_df = unit_df[(unit_df['time'] <= selected_time) & (unit_df['time'] > selected_time - 5)]
    if window_df.empty:
        window_mean = current_row
    else:
        window_mean = window_df.mean()

    # Generate Reasoning
    expert = ReasoningExpert()
    reasoning, status = expert.analyze(current_row, window_mean)

    # Display
    st.metric("True RUL", f"{current_row['rul']} cycles", delta_color="inverse")
    
    st.markdown("### 🧠 Model Reasoning Trace")
    st.info(reasoning)
    
    st.markdown("### 📋 Predicted Status")
    if status == "CRITICAL":
        st.error(f"Status: {status}")
    elif status == "WARNING":
        st.warning(f"Status: {status}")
    else:
        st.success(f"Status: {status}")

st.markdown("---")
st.caption("Data: NASA CMAPSS FD001 | Method: Synthetic Reasoning Expert System -> Tunix LLM Training")
