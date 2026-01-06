# Project Title: TurbofanReason: Teaching LLMs to Diagnose Engine Failure with Tunix

## Subtitle
A JAX-native approach to "Show Your Work" in Predictive Maintenance using Generated Reasoning Traces.

---

## 1. Overview
In high-stakes domains like aviation maintenance, a "black box" prediction is insufficient. An AI that predicts a jet engine has 15 cycles left before failure must explain *why*—citing specific sensor trends, pressure variances, and heat signatures.

**TurbofanReason** is a fine-tuned Gemma 2B model that solves this transparency gap. Using Google Tunix and JAX, we trained the model not just to predict Remaining Useful Life (RUL), but to generate a step-by-step engineering diagnosis ("Reasoning Trace") preceding the answer.

## 2. The Challenge
Standard open-source datasets for predictive maintenance (like NASA CMAPSS) contain only raw numbers, not the text-based reasoning required to train a "Chain of Thought" (CoT) model.
*   **Input**: `[Cycle 140: Sensor_3=1500, Sensor_4=1420...]`
*   **Target**: `RUL: 50`

To win this challenge, we had to bridge the gap between *numerical time-series* and *linguistic reasoning*.

## 3. Methodology: Synthetic Reasoning
We introduced a **Reasoning Synthesis Layer** (RSL). Before training the LLM, we implemented a rule-based expert system (the "Teacher") that analyzes the raw NASA sensor data and generates a natural language valid ground truth.

### The Pipeline
1.  **Data Ingestion**: We load the NASA CMAPSS `train_FD001` dataset.
2.  **Analysis**: The RSL scans for specific failure signatures:
    *   *High Pressure Compressor (HPC) Outlet Temperature* drift.
    *   *Static Pressure* restriction.
3.  **Synthesis**: The RSL generates a training target:
    ```xml
    <reasoning>
    LPT Outlet Temp (Sensor 4) is 1420, exceeding nominal range. 
    Static Pressure is diverging. Multiple failure signatures detected.
    </reasoning>
    <answer>Status: WARNING | EST_RUL: 72</answer>
    ```
4.  **Tunix Fine-Tuning**: We fine-tuned **Gemma 2B** on this synthetic dataset using Tunix on Cloud TPU v5e. The model learns to map the numerical patterns (Input) to the reasoning structure (Output).

## 4. Why Tunix & JAX?
We utilized Tunix for its optimized JAX backend, allowing efficient fine-tuning of Gemma on TPU hardware.
*   **Flexibility**: Tunix allowed us to define a custom data formatting pipeline that seamlessly integrated the `<reasoning>` XML tags.
*   **Speed**: The JAX execution path significantly reduced iteration time compared to standard PyTorch loops on the provided TPU resources.
*   **RL Post-Training**: The structure allows for Reinforcement Learning (RL) fine-tuning where the reward function `R(x)` validates both the structural integrity of the XML tags and the numeric accuracy of the RUL prediction.

## 5. Results & Impact
The resulting model demonstrates that 2B-parameter models can effectively "show their work" even in non-native domains like time-series analysis. By forcing the model to articulate the *cause* (sensor drift) before the *effect* (maintenance alert), we achieve:
1.  **Trust**: Operators can verify the model's logic.
2.  **Accuracy**: The Chain-of-Thought process stabilizes the final prediction.

