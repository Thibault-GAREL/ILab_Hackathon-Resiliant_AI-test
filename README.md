# 🤖 Gemma 4 E4B Model Compression — Resilient AI Challenge

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-%2376B900.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

---

<p align="center">
  <img src="https://github.com/Thibault-GAREL/Thibault-GAREL/raw/main/Logo_Group_Projects/group_resilient_sq.png" alt="Resilient AI Challenge Logo" width="150">
</p>

---

## 📝 Project Description

This project is part of the **Resilient AI Challenge**, a France-India-UNESCO collaboration challenging teams to compress state-of-the-art models while maintaining performance. We focus on compressing **Google's Gemma 4 E4B** for the **Image-to-Text task**. The goal is to reduce model size and latency without sacrificing accuracy—essential for deploying AI on resource-constrained devices.

This work is conducted at **Intelligence Lab** (ECE Paris), where we combine research, prototyping, and innovation to solve real-world AI challenges.

---

## ⚙️ Features

  🎯 Target **Gemma 4 E4B** model compression for image-to-text tasks

  📉 Implement quantization and pruning techniques to reduce model size

  ⚡ Optimize inference latency on edge devices

  📊 Track experiments with **MLflow** for reproducibility

  🧪 Comprehensive validation metrics (BLEU, CIDEr, METEOR)

  🔧 Modular architecture: easy to swap compression techniques

  🌍 Collaborative framework supporting multi-region teams

---

## 📂 Repository structure

```bash
├── src/                           # Source code
│   ├── models/
│   │   ├── model.py              # Model architecture
│   │   └── train.py              # Training loop with MLflow
│   ├── data/
│   │   └── make_dataset.py        # Dataset preparation
│   ├── validation/
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── test_infer_gemma4.py  # Inference testing
│   ├── config.py                 # Configuration (Pydantic)
│   └── utils.py                  # Utilities
│
├── tests/
│   └── visualization/
│       └── model_viewer.py        # Model inspection tools
│
├── data/
│   ├── 1-raw/                    # Original datasets
│   ├── 2-processed/              # Preprocessed data
│   └── 3-external/               # External resources
│
├── outputs/
│   ├── models/                   # Compressed models
│   ├── logs/                     # Training logs
│   └── results/                  # Evaluation results
│
├── .env                          # Environment variables
├── mlflow.db                     # MLflow experiment tracking
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC

Clone the repository and set up the environment:

```bash
git clone https://github.com/Thibault-GAREL/ILab_Hackathon-Resiliant_AI-test.git
cd ILab_Hackathon-Resiliant_AI-test

# Create a virtual environment
python -m venv .venv  # if you don't have a virtual environment
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

⚠️ You need a **CUDA-compatible GPU** (NVIDIA recommended) and sufficient VRAM (~6-24 GB depending on model size).

### Train the model

```bash
python -m src.models.train
```

### Validate / Evaluate

```bash
python -m src.validation.test_infer_gemma4
```

### Track experiments with MLflow

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

---

## 👥 Team Members

<p align="center">
  <a href="https://github.com/Thibault-GAREL"><img src="https://img.shields.io/badge/Thibault_Garel-181717?style=flat-square&logo=github&logoColor=white"/></a>
  <a href="https://github.com/axelbrons"><img src="https://img.shields.io/badge/Axel_Br%C3%B6ns-181717?style=flat-square&logo=github&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/mathis-lacombe34/"><img src="https://img.shields.io/badge/Mathis_Lacombe-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/jarfino-houngbadji/"><img src="https://img.shields.io/badge/Jarfino_Houngbadji-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/demba-sow-achta/"><img src="https://img.shields.io/badge/Achta_Sow_Demba-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/djebril-laouedj-9684b4219/"><img src="https://img.shields.io/badge/Djebril_Laouedj-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/bikram-pratim-bhuyan-01887589/"><img src="https://img.shields.io/badge/Bikram_Bhuyan-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
</p>

---

## 📖 Inspiration / Sources

This project is part of the **Sustainable AI Coalition's Resilient AI Challenge**:
- 🌍 [Resilient AI Challenge](https://www.sustainableaicoalition.org/resilient-ai-challenge/)
- 🎓 [Intelligence Lab — ECE Paris](https://www.ece.fr/intelligence-lab/)
- 📄 [Gemma 4 E4B Model](https://ai.google.dev/gemma)

Code created by the **Resilient AI Team** 😎 — [Intelligence Lab](https://www.ece.fr/intelligence-lab/)

---

⚠️ **Note:** This project is **work in progress** 🚨. Compression techniques are still being evaluated and optimized.
