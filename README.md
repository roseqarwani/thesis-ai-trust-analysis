# Thesis AI Trust Analysis

This repository contains the Python analysis workflow for my bachelor thesis on reasoning, AI correctness, trust and reliance in user-facing LLM systems.

## Project structure

- `data/raw/`: raw Qualtrics CSV files, not committed to GitHub
- `data/processed/`: cleaned long-format datasets, not committed to GitHub
- `notebooks/`: Google Colab / Jupyter notebooks
- `src/`: reusable Python scripts
- `outputs/tables/`: exported descriptive and model tables
- `outputs/figures/`: exported graphs

## Main analysis steps

1. Load raw Qualtrics CSV
2. Clean metadata and incomplete responses
3. Convert from wide format to long format
4. Create variables:
   - reasoning
   - ai_correctness
   - trust_score
   - reliance_score
   - confidence_change
   - changed_answer
   - changed_to_ai_answer
5. Run descriptive analyses
6. Run regression and mixed-effects models
7. Export tables and figures
