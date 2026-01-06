# Google Tunix Hackathon: NASA Turbofan Reasoning

This project implements a predictive maintenance model that "shows its work". We train a Gemma 2B model to analyze NASA jet engine sensor data and output a step-by-step reasoning trace explaining the machine status before predicting the Remaining Useful Life (RUL).

## Files
- `nasa_turbofan_colab.py`: **Main Script**. Contains the Data Loader (NASA CMAPSS), Synthetic Reasoning Engine, and Tunix (JAX) Training Loop. This is designed to be copy-pasted into Google Colab.
- `submission.ipynb`: The Jupyter Notebook generated from the script above, ready for Kaggle submission.
- `app.py`: **Interactive Visualizer**. A Streamlit app to explore the data and see the "Reasoning Synthesis" in real-time.
- `kaggle_writeup.md`: The official writeup for the hackathon entry, detailing the methodology, "Reasoning Synthesis" layer, and results.
- `export_to_notebook.py`: Utility script to regenerate `submission.ipynb` from the colab script.

## How to Run

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook.
3. Set Runtime to **TPU v2** or **TPU v3**.
4. Copy the contents of `nasa_turbofan_colab.py` into a cell.
5. Run.

### Option 2: Visualization (Local)
To see the "Reasoning Synthesis" layer in action:
```bash
pip install streamlit plotly pandas requests
streamlit run app.py
```
This launches a web dashboard where you can scrub through engine life-cycles and see the generated explanation change from "Healthy" to "Critical".

### Option 3: Kaggle
1. Upload `submission.ipynb` to your Kaggle project.
2. Select TPU v5e accelerator.
3. Run.

## Methodology
The core innovation is the **Reasoning Synthesis** layer. Since the raw NASA dataset contains only numbers, we implemented a rule-based expert system that analyzes sensor trends (e.g., *HPC Outlet Temp drifting*) and generates a natural language explanation. We use this synthetic data to fine-tune the LLM to output:
```xml
<reasoning> Sensor 4 is exceeding nominal range... </reasoning>
<answer> RUL: 45 cycles </answer>
```
