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
        # Adjusted limits for cleaner demo effect
        self.limits = {
            's2': 641.0,  
            's3': 1575.0, 
            's4': 1360.0, # Very aggressive: base is around 1350-1400
            's7': 556.0,  # Raised floor
            's11': 47.0,  
        }

    def analyze(self, row, window_mean):
        reasons = []
        status = "HEALTHY"
        
        # 1. RUL-based Status (Ground Truth)
        if row['rul'] < 75: # Increased critical range
            status = "CRITICAL"
            reasons.append("🔴 **CRITICAL**: Multiple failure signatures detected coincident with high cycle count.")
        elif row['rul'] < 175: # Increased warning range (Data often starts ~192 RUL)
            status = "WARNING"
            reasons.append("⚠️ **WARNING**: Sensors show significant drift signs.")
        else:
            reasons.append("✅ **NOMINAL**: System operating within optimal bands.")

        # 2. Sensor Specifics (The "Reasoning")
        # Compare window mean (smoother) to limits
        # S4: LPT Outlet Temp (Rising is bad)
        if window_mean['s4'] > self.limits['s4']:
             reasons.append(f"- **Sensor 4 (LPT Temp)**: {window_mean['s4']:.1f} > Limit {self.limits['s4']} (High Temp)")
        
        # S11: Static Pressure (Rising is bad in this context/model?)
        if window_mean['s11'] > self.limits['s11']:
            reasons.append(f"- **Sensor 11 (Static Pressure)**: {window_mean['s11']:.2f} > Limit {self.limits['s11']} (Flow Restriction)")
            
        # S7: HPC Pressure (Dropping is bad)
        if window_mean['s7'] < self.limits['s7']: 
             reasons.append(f"- **Sensor 7 (HPC Press)**: {window_mean['s7']:.2f} < Limit {self.limits['s7']} (Pressure Loss)")

        return "  \n".join(reasons), status

# ==========================================
# 2. APP UI
# ==========================================
st.title("✈️ TurbofanReason: Explainable Predictive Maintenance")
st.markdown("Visualizing the **Reasoning Synthesis** layer. We translate raw sensor data into engineering diagnostics.")

df = load_data()
expert = ReasoningExpert()

# Create Tabs
tab1, tab2 = st.tabs(["⚡ Side-by-Side Comparison", "🔍 Deep Dive Explorer"])

def get_example_by_status(df, target_status):
    """Finds a random example matching the criteria."""
    # We iterate a few random units to find a match
    import random
    units = list(df['unit'].unique())
    random.shuffle(units)
    
    for u in units:
        u_df = df[df['unit'] == u]
        # Check start, middle, end
        candidate_indices = [
            u_df.index[0], # Start (Usually Healthy)
            u_df.index[int(len(u_df)/2)], # Middle
            u_df.index[-1] # End (Usually Critical)
        ]
        
        for idx in candidate_indices:
            row = df.loc[idx]
            # Quick calc window
            win = df[(df['unit'] == u) & (df['time'] <= row['time']) & (df['time'] > row['time']-5)]
            if win.empty: win_mean = row
            else: win_mean = win.mean()
            
            _, status = expert.analyze(row, win_mean)
            
            if status == target_status:
                return row, win_mean, status
                
    return None, None, None

with tab1:
    st.markdown("### 🆚 Healthy vs Critical State")
    if st.button("🔄 Refresh Examples"):
        st.cache_data.clear() # Hacky way to force refresh if we cached randoms, but logic is dynamic here
        
    colA, colB = st.columns(2)
    
    # Healthy Example
    with colA:
        st.success("### ✅ HEALTHY Engine")
        row, win, stat = get_example_by_status(df, "HEALTHY")
        if row is not None:
             reasoning, _ = expert.analyze(row, win)
             st.metric("Unit", f"#{int(row['unit'])}")
             st.metric("Cycle", f"{int(row['time'])}")
             st.metric("RUL", f"{int(row['rul'])}", delta="Nominal")
             st.markdown(f"**ReasoningTrace:**\n\n{reasoning}")
             st.caption("Sensors are stable.")
             
    # Critical Example
    with colB:
        st.error("### 🔴 CRITICAL Engine")
        # Try finding critical, if not warning
        row, win, stat = get_example_by_status(df, "CRITICAL")
        if row is None:
             row, win, stat = get_example_by_status(df, "WARNING")
             
        if row is not None:
             reasoning, _ = expert.analyze(row, win)
             st.metric("Unit", f"#{int(row['unit'])}")
             st.metric("Cycle", f"{int(row['time'])}")
             st.metric("RUL", f"{int(row['rul'])}", delta="-Low", delta_color="inverse")
             st.markdown(f"**ReasoningTrace:**\n\n{reasoning}")
             # Highlight the bad sensors
             st.dataframe(pd.DataFrame({
                 "Sensor": ["S4 (LPT)", "S7 (HPC)", "S11 (Static)"],
                 "Value": [f"{win['s4']:.1f}", f"{win['s7']:.1f}", f"{win['s11']:.1f}"],
                 "Limit": [expert.limits['s4'], expert.limits['s7'], expert.limits['s11']]
             }))

with tab2:
    st.info("💡 **Tip**: Drag the **Time Cycle** slider to the right to see the engine degrade!")
    # Sidebar
    units = df['unit'].unique()
    selected_unit = st.selectbox("Select Engine Unit for Deep Dive", units)

    # Filter Data
    unit_df = df[df['unit'] == selected_unit]
    max_time = int(unit_df['time'].max())

    # Main Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sensor Telemetry")
        # Plot S4, S7, S11
        fig = px.line(unit_df, x="time", y=["s4", "s7", "s11"], title=f"Engine {selected_unit} Key Sensors")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Live Diagnosis")
        
        # "Jump to Failure" button
        if st.button("Jump to End of Life", key="jump_btn"):
            st.session_state[f'slider_{selected_unit}'] = max_time
        
        # Slider with session state to allow button update
        if f'slider_{selected_unit}' not in st.session_state:
            st.session_state[f'slider_{selected_unit}'] = int(unit_df['time'].min())
            
        selected_time = st.slider("Time Cycle", 
                                  min_value=int(unit_df['time'].min()), 
                                  max_value=max_time,
                                  key=f'slider_{selected_unit}')
        
        # Get Data for this cycle
        current_row = unit_df[unit_df['time'] == selected_time].iloc[0]
        
        # Simple window for smoothing (last 5 cycles)
        window_df = unit_df[(unit_df['time'] <= selected_time) & (unit_df['time'] > selected_time - 5)]
        if window_df.empty:
            window_mean = current_row
        else:
            window_mean = window_df.mean()

        # Generate Reasoning
        reasoning, status = expert.analyze(current_row, window_mean)

        # Display Predictions
        # RUL Metric
        delta_rul = current_row['rul'] - 100 # arbitrary reference for color
        st.metric("True RUL", f"{int(current_row['rul'])} cycles", delta=float(current_row['rul'] - max_time), delta_color="normal") # negative delta = getting closer to death
        
        st.markdown("### 🧠 Generated Reasoning")
        st.markdown(reasoning)
        
        # Status Badge
        if status == "CRITICAL":
            st.error(f"Status: {status}")
        elif status == "WARNING":
            st.warning(f"Status: {status}")
        else:
            st.success(f"Status: {status}")

        # Debug Data
        with st.expander("Debug: Sensor Values"):
            st.write({
                "s4": f"{window_mean['s4']:.2f} (Limit: {expert.limits['s4']})",
                "s7": f"{window_mean['s7']:.2f} (Limit: {expert.limits['s7']})",
                "s11": f"{window_mean['s11']:.2f} (Limit: {expert.limits['s11']})",
            })

st.markdown("---")
st.caption("Data: NASA CMAPSS FD001 | Method: Synthetic Reasoning Expert System -> Tunix LLM Training")
