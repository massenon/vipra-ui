# ViPRA-UI: Automated UI-User Review Mismatch Detection

![ViPRA-UI Demo](./path/to/your/demo_screenshot.gif) 
<!-- Create a small GIF of your app working and add it here -->

This repository contains the official implementation of the **ViPRA-UI** framework, a tool for the automated detection of UI-User Review Mismatches in mobile applications. ViPRA-UI leverages Multimodal Large Language Models (MLLMs) to analyze GUI screenshots and user reviews, identifying discrepancies that often indicate software defects.

This tool is the practical implementation of the research presented in our paper:
> **[Your Paper Title]** - [Link to your paper or arXiv preprint]

---

## ✨ Features

-   **Cross-Modal Analysis:** Compares visual GUI screenshots with textual user feedback to find inconsistencies.
-   **Structural UI Understanding:** Parses XML View Hierarchies to understand the properties of UI elements (e.g., `clickable=false`).
-   **Visual Grounding:** Automatically annotates screenshots to visually link user complaints to specific UI elements.
-   **MLLM-Powered Reasoning:** Uses a structured vision prompt to guide a powerful MLLM (`microsoft/Florence-2-large`) through a logical analysis.
-   **Interactive Demo:** A user-friendly Gradio web interface for easy testing and demonstration.

## 🚀 Live Demo on Hugging Face Spaces

You can try ViPRA-UI live without any installation:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/massenon/vipra-ui) 
<!-- This link will be updated after deployment -->

## 🛠️ Setup and Installation (Local)

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (recommended for performance)
- An environment with sufficient VRAM for the MLLM (e.g., >8GB)

### Installation Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/massenon/vipra-ui.git
    cd vipra-ui
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## ▶️ How to Run

1.  **Configure the Model:**
    Open `config.yaml` and ensure the `device` is set correctly (`cuda` or `cpu`).

2.  **Launch the Gradio Demo:**
    ```bash
    python app.py
    ```
    Navigate to the local URL provided in your terminal (e.g., `http://127.0.0.1:7860`).

## 🏛️ Project Structure

```
vipra-ui/
├── app.py                 # Main Gradio application
├── core/                  # Core logic modules
├── data/                  # Placeholder for data
├── examples/              # Example files for the demo
├── config.yaml            # System configuration
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).