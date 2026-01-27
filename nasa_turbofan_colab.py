!pip install -q transformers datasets peft accelerate bitsandbytes
!pip install -q torch trl pandas numpy requests
!pip install -q sentencepiece protobuf

import os
from huggingface_hub import login

# 1. Force-remove the conflicting secret from the environment
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]

print("Old credentials cleared.")
print("Please paste your NEW token below when prompted:")

# 2. Force a manual login prompt
# This bypasses the Colab Vault entirely
token_input = input("Paste your Hugging Face Token here: ").strip()
login(token=token_input)

print("✅ Login command sent. Proceeding to training...")

# ===========================================================
# NASA Turbofan Reasoning: JAX/TPU Version (Gemma 2 - Manual Download Fix)
# ===========================================================
#
# STRATEGY: We manually download the model weights first to bypass
# the KerasNLP authentication bug.
#

import os
import sys
import subprocess
import getpass

# -----------------------------------------------------------
# STEP 1: INSTALL
# -----------------------------------------------------------
print("Installing JAX, KerasNLP, and Hugging Face Hub...")
packages = [
    "keras-nlp",
    "keras>=3.0.0",
    "huggingface_hub",
    "pandas",
    "numpy",
    "requests"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

# -----------------------------------------------------------
# STEP 2: AUTHENTICATION (Environment Injection)
# -----------------------------------------------------------
from huggingface_hub import login, snapshot_download

print("\n" + "="*50)
print("AUTHENTICATION REQUIRED")
print("Paste your Hugging Face token (Read permission) below.")
print("="*50)

token_input = getpass.getpass("Hugging Face Token: ").strip()

# FIX: We inject the token into the OS environment so ALL libraries see it
os.environ["HF_TOKEN"] = token_input
login(token=token_input)

# -----------------------------------------------------------
# STEP 3: PRE-EMPTIVE DOWNLOAD (The Fix)
# -----------------------------------------------------------
print("\n" + "="*50)
print("MANUALLY DOWNLOADING GEMMA 2")
print("This bypasses the '401 Unauthorized' error from Keras.")
print("="*50)

try:
    # We download the model to a local folder first
    # This confirms your token works and caches the files
    local_model_path = snapshot_download(
        repo_id="google/gemma-2-2b",
        token=token_input,
        ignore_patterns=["*.msgpack", "*.h5", "*.tflite"] # Only get Safetensors/JAX weights
    )
    print(f"\n✅ Download complete! Files saved to: {local_model_path}")
except Exception as e:
    print(f"\n❌ DOWNLOAD FAILED: {e}")
    print("STOP HERE. If this fails, your token is invalid or you have not accepted the license at https://huggingface.co/google/gemma-2-2b")
    raise e

# -----------------------------------------------------------
# STEP 4: JAX SETUP
# -----------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import keras
import keras_nlp
import jax
import numpy as np
import pandas as pd
import requests
import zipfile
from pathlib import Path

keras.config.set_floatx("bfloat16")
print(f"\nTPU Devices: {jax.devices()}")

# -----------------------------------------------------------
# STEP 5: PREPARE DATA
# -----------------------------------------------------------
CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
WINDOW_SIZE = 30

def get_data():
    cache_dir = Path("./cmapss")
    cache_dir.mkdir(exist_ok=True)
    zip_path = cache_dir / "CMAPSSData.zip"

    if not zip_path.exists():
        print("Downloading NASA dataset...")
        try:
            r = requests.get(CMAPSS_URL, timeout=30)
            zip_path.write_bytes(r.content)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(cache_dir)
        except:
            return pd.DataFrame(np.random.randn(500, 26))

    txt_files = list(cache_dir.glob("**/train_FD001.txt"))
    if txt_files:
        cols = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        return pd.read_csv(txt_files[0], sep=r'\s+', header=None, names=cols)
    return pd.DataFrame()

class ReasoningExpert:
    KEY_SENSORS = ['s2', 's3', 's4', 's7', 's11', 's12']
    def generate_trace(self, window_df, rul, unit_id, cycle):
        observations = []
        for sensor in self.KEY_SENSORS:
            if sensor in window_df.columns:
                vals = window_df[sensor].values
                slope = np.polyfit(range(len(vals)), vals, 1)[0]
                if slope > 0.001: observations.append(f"{sensor} ↑")
                elif slope < -0.001: observations.append(f"{sensor} ↓")

        if rul < 30: s, r, a = "CRITICAL", "HIGH", "Immediate Maint"
        elif rul < 75: s, r, a = "WARNING", "MEDIUM", "Schedule Maint"
        else: s, r, a = "HEALTHY", "LOW", "Normal Ops"
        obs_text = ", ".join(observations[:3]) if observations else "Stable"

        return f"""Unit: {unit_id}, Cycle: {cycle}
Sensors: {window_df[self.KEY_SENSORS].iloc[-1].to_dict()}

<reasoning>
Analysis: {obs_text}.
Logic: Degradation slope indicates wear.
Prediction: RUL ≈ {rul}.
</reasoning>
<answer>
Status: {s}
Risk: {r}
Action: {a}
</answer>"""

print("Preparing training data...")
df = get_data()
sensor_cols = [c for c in df.columns if c.startswith('s')]
for col in sensor_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

max_cycles = df.groupby('unit')['time'].max().rename('max_time')
df = df.merge(max_cycles, on='unit')
df['rul'] = df['max_time'] - df['time']

expert = ReasoningExpert()
train_data = []
for unit in df['unit'].unique()[:15]:
    u_data = df[df['unit'] == unit].reset_index(drop=True)
    for i in range(WINDOW_SIZE, len(u_data), 5):
        win = u_data.iloc[i-WINDOW_SIZE:i]
        row = u_data.iloc[i-1]
        train_data.append(expert.generate_trace(win, int(row['rul']), unit, int(row['time'])))

print(f"Generated {len(train_data)} reasoning examples.")

# -----------------------------------------------------------
# STEP 6: LOAD & TRAIN (Using Local Cache)
# -----------------------------------------------------------
print("\nLoading Gemma 2 (2B) from local cache...")
# NOTE: We still use the preset string, but because we ran snapshot_download,
# Keras will find the files in the HuggingFace cache immediately.
gemma = keras_nlp.models.GemmaCausalLM.from_preset("hf://google/gemma-2-2b")

print("Enabling LoRA (Rank=8)...")
gemma.backbone.enable_lora(rank=8)

print("Compiling XLA Graph...")
gemma.preprocessor.sequence_length = 256
gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01),
    weighted_metrics=["accuracy"],
)

print("\nSTARTING TRAINING (Step 1 takes ~60s)...")
gemma.fit(train_data, batch_size=4, epochs=1)

# -----------------------------------------------------------
# STEP 7: INFERENCE
# -----------------------------------------------------------
print("\n" + "="*50)
print("TESTING MODEL")
print("="*50)
test_prompt = """Unit: 99, Cycle: 150
Sensors: {'s2': 0.8, 's3': 0.5, 's4': 1.2}

<reasoning>"""

output = gemma.generate(test_prompt, max_length=200)
print(output)

