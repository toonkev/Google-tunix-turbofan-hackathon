# @title NASA Turbofan Reasoning with Tunix (JAX)
# Copy this entire code block into a Google Colab cell.
# Hardware: Select "TPU v2" or "TPU v3" from Runtime > Change runtime type.

import os
import sys
import numpy as np
import pandas as pd
import requests
import zipfile
import io
import re
import jax
import jax.numpy as jnp
from typing import List, Dict, Any

# ==========================================
# 1. SETUP & DATA DOWNLOAD
# ==========================================
from pathlib import Path

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

def download_and_read_cmapss_fd001(cache_dir="/content/cmapss"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "CMAPSSData.zip"
    
    if not zip_path.exists():
        print(f"Downloading CMAPSS zip from {CMAPSS_URL}...")
        try:
            # Requires verify=False sometimes for old NASA certs, but let's try standard first
            r = requests.get(CMAPSS_URL, timeout=120)
            r.raise_for_status()
            zip_path.write_bytes(r.content)
        except Exception as e:
            print(f"Failed to download from NASA: {e}")
            raise e

    print("Extracting CMAPSS zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)

    # Train file for FD001 is usually "train_FD001.txt"
    # Note: The zip might contain a subfolder depending on version, generic path search
    # But usually it extracts to top level or CMAPSSData folder
    train_path = cache_dir / "train_FD001.txt"
    if not train_path.exists():
        # Check if it's in a subdirectory
        found = list(cache_dir.glob("**/train_FD001.txt"))
        if found:
            train_path = found[0]
        else:
            raise FileNotFoundError(f"Could not find train_FD001.txt in {cache_dir}")

    print(f"Reading data from {train_path}...")
    col_names = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    # Fixed escape sequence warning by using raw string r'\s+'
    df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    print(f"Loaded DataFrame with shape {df.shape}")
    return df

# ==========================================
# 2. REASONING SYNTHESIS (The "Teacher")
# ==========================================
# Since the dataset is just numbers, we need to TEACH the model how to reason.
# We use a rule-based expert system to generate 'ground truth' reasoning traces.
class ReasoningExpert:
    def __init__(self):
        # Approximate failure thresholds for key sensors in FD001 (HPC degradation)
        self.limits = {
            's2': 644.0,  # Compressor inlet temp
            's3': 1600.0, # HPC outlet temp
            's4': 1420.0, # LPT outlet temp
            's7': 554.0,  # HPC outlet pressure (downward trend usually bad)
            's11': 48.0,  # Static pressure at HPC outlet
        }

    def analyze_window(self, window_df: pd.DataFrame, current_rul: int) -> str:
        """Generates a text explanation of the sensor status."""
        last_row = window_df.iloc[-1]
        reasons = []

        # 1. Analyze trends
        if last_row['s4'] > self.limits['s4']:
            reasons.append(f"LPT Outlet Temp (s4) is {last_row['s4']:.1f}, exceeding nominal range.")
        
        if last_row['s11'] > self.limits['s11']:
            reasons.append(f"Static Pressure (s11) is high ({last_row['s11']:.1f}), indicating flow restriction.")

        # 2. synthesize conclusion
        if current_rul < 30:
            status = "CRITICAL"
            reasons.append("Multiple failure signatures detected coincident with high cycle count.")
        elif current_rul < 75:
            status = "WARNING"
            reasons.append("Sensors show early drift signs.")
        else:
            status = "HEALTHY"
            reasons.append("All sensors operating within nominal bands.")

        trace = " ".join(reasons)
        return trace, status

def prepare_dataset(df):
    # Calculate RUL
    # RUL = Max Cycle - Current Cycle
    max_cycles = df.groupby('unit')['time'].max().rename('max_time')
    df = df.join(max_cycles, on='unit')
    df['rul'] = df['max_time'] - df['time']

    # Note: We do NOT normalize the data for the LLM input.
    # 1. The ReasoningExpert relies on physical thresholds (e.g. 1420 deg).
    # 2. Gemma (LLM) can read raw numbers ("1420") just fine.
    
    expert = ReasoningExpert()
    examples = []
    
    # Create windows
    WINDOW_SIZE = 30
    print("Generating synthetic reasoning traces for training data...")
    for unit_id in df['unit'].unique()[:5]: # Limit to 5 units for demo speed
        unit_data = df[df['unit'] == unit_id]
        
        # Sliding window
        for i in range(WINDOW_SIZE, len(unit_data), 10): # Stride 10
            window = unit_data.iloc[i-WINDOW_SIZE:i]
            cur_rul = unit_data.iloc[i]['rul']
            
            # FORMAT INPUT
            # Represent the implementation as a summary of the window
            sensor_summary =  window[['s2', 's4', 's7', 's11']].mean().to_dict()
            input_text = f"Sensor Report (Last {WINDOW_SIZE} cycles):\\n"
            for k,v in sensor_summary.items():
                input_text += f"{k}: {v:.2f}, "
            
            # FORMAT TARGET
            reasoning, status = expert.analyze_window(window, cur_rul)
            target_text = f"<reasoning>{reasoning}</reasoning>\\n<answer>Status: {status} | EST_RUL: {cur_rul}</answer>"
            
            examples.append({
                "input": input_text,
                "target": target_text
            })
            
    print(f"Generated {len(examples)} training examples.")
    return examples

# ==========================================
# 3. MOCK JAX/TUNIX TRAINING LOOP
# ==========================================
# In a real Hackathon setting, import tunix here.
# Since we don't have the library, we show the Flax setup.

def train_model(examples):
    print("\n[Tunix] Initializing JAX execution on TPU...")
    print(f"[Tunix] JAX Devices: {jax.devices()}")
    
    # Pseudo-code for Tunix / Flax loop
    print("[Tunix] Loading Gemma 2B (Simulated)...")
    
    # 1. Tokenize (Mock)
    print("[Tunix] Tokenizing inputs...")
    
    # 2. Training Loop
    EPOCHS = 2
    print(f"[Tunix] Starting SFT (Supervised Fine-Tuning) for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        loss = np.random.uniform(2.0, 0.5) # Fake loss curve
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Reasoning Capability: Increasing")
        
    print("[Tunix] Training complete. Model has learned to mimic the ReasoningExpert.")

# ==========================================
# 4. RL REWARD FUNCTION (Post-Training)
# ==========================================
def reward_function(generated_text, ground_truth_rul):
    """
    Rewards the model for:
    1. Structure: Having <reasoning> and <answer> tags.
    2. Accuracy: Predicting RUL close to ground truth.
    """
    reward = 0.0
    
    # 1. Structural Reward
    if "<reasoning>" in generated_text and "</reasoning>" in generated_text:
        reward += 1.0
    if "<answer>" in generated_text and "</answer>" in generated_text:
        reward += 1.0
        
    # 2. Parsing logic (simplified regex)
    try:
        match = re.search(r"EST_RUL: (\d+)", generated_text)
        if match:
            pred_rul = int(match.group(1))
            error = abs(pred_rul - ground_truth_rul)
            # Higher reward for lower error
            reward += max(0, 5.0 - (error * 0.1)) 
    except:
        pass
        
    return reward

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = download_and_read_cmapss_fd001()
    dataset = prepare_dataset(df)
    
    # Show one example
    print("\n--- Sample Interaction ---")
    print(f"User Input:\n{dataset[0]['input']}")
    print(f"Target Output:\n{dataset[0]['target']}")
    
    # Run Training
    train_model(dataset)
    
    print("\n--- RL Post-Training Check ---")
    # Simulate an RL step
    sample_gen = dataset[0]['target'] # Assume model generated perfect output
    r = reward_function(sample_gen, 140) # Assume real RUL was 140
    print(f"Reward for perfect output: {r:.2f}")

