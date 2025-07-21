# Robust-Estimation-and-Recommendation-System-with-Data-Quality-Automation
## 📌 Overview

This project implements an end-to-end machine learning pipeline for **estimation and recommendation tasks**, following the CRISP-DM framework. It includes:

- Automated data quality checks (missing values, duplicates, value ranges, etc.)
- Data leakage detection and mitigation
- Estimation of housing prices using real-world data
- Recommendation modeling to predict fantasy book ratings
- Training at scale using HPC (High Performance Computing) via SLURM job submission

The solution was developed as part of CSC373/673: Data Mining at Wake Forest University (Spring 2025).

---

## 🧠 Methodology: CRISP-DM Pipeline

This project is structured into the following phases:

| Phase | Task |
|-------|------|
| **II** | Data understanding & quality checking |
| **III** | Data preparation & modeling (housing + books) |
| **IV** | Evaluation and performance improvement |
| **V** | HPC training with large-scale fantasy book reviews |

---

## 📂 Project Structure

assignment_2/
│
├── data/ # 📁 Raw data files
│ └── feature_descriptions.csv
│
├── output/ # 📁 Reports and model results
│ ├── data_quality_report.csv
│ ├── fantasy_modeling_report.txt
│ └── fantasy_model_report(part4).txt
│
├── scripts/ # 📁 All Python scripts
│ ├── init.py
│ ├── check_quality.py # Phase II: Data quality + leakage check
│ ├── transformers.py # Phase III: Custom Transformer class
│ ├── utils.py # Phase IV: Feature engineering utils
│ └── assignment_2.py # Main execution script
│
└── README.md # 📘 This file
