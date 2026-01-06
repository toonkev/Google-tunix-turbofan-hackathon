import json
import os

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def create_notebook():
    # 1. Read source files
    # We only need the single Colab script now
    nasa_code = read_file('nasa_turbofan_colab.py')

    # 3. Define Notebook Structure
    cells = []

    # Cell 1: Intro Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Google Tunix Hackathon: NASA Turbofan Reasoning\n",
            "This notebook implements the **TurbofanReason** pipeline.\n",
            "1. Downloads NASA CMAPSS Data.\n",
            "2. Synthesizes <reasoning> traces using an Expert System.\n",
            "3. Trains Gemma 2B using Tunix (JAX)."
        ]
    })

    # Cell 2: The Main Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": nasa_code.splitlines(keepends=True)
    })

    # 4. Construct JSON
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # 5. Save
    output_path = 'submission.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Generated {output_path} successfully using nasa_turbofan_colab.py.")

if __name__ == "__main__":
    create_notebook()
