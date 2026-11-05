# AI-Driven Health Expenditure Prediction

Machine-learning models and an interactive what-if scenario tool for modeling and forecasting health-expenditure trends. Developed during a mandatory summer internship at the Social Security Institution (SGK), Health Software Department (2025).

> **Note:** This is an intern project built entirely on **synthetic data**. Interns are not given access to real or sensitive data (per KVKK — Turkey's data-protection law — and institutional confidentiality), so all datasets were generated to approximate realistic distributions. It is not an official SGK product and uses no real health, prescription or cost data.

## Problem

Health expenditure is one of the largest items in public budgets, and SGK finances health costs for millions of people. Forecasting these expenditures is hard because they depend on demographics, disease incidence, service demand, drug prices and health inflation. This project explores whether ML can model those expenditures and give decision-makers a tool to test budget scenarios proactively.

## Data (synthetic)

Three complementary synthetic datasets covering 2015–2023 (monthly):

- **Service-type based** (outpatient / inpatient / surgery)
- **Institution based** (public / private / university hospitals)
- **Prescription based** (branded / generic drugs)

Generation reflected historical trends, monthly seasonality, annual health-inflation rates and observational noise, with ~100 observations per year-month combination and categorical segmentation.

## Methods

- **Preprocessing** — missing-value handling, outlier detection, type conversion, scaling/normalization.
- **EDA** — distributions (histograms, boxplots), correlation matrices, cost comparisons by service/institution/prescription.
- **Modeling** — Linear, Ridge, Lasso, Random Forest, Gradient Boosting and MLP regression; 80/20 split; evaluated with MSE, RMSE and R²; k-fold cross-validation.
- **Overfitting control** — learning curves, bias–variance and permutation-importance analysis.
- **What-if module** — an interactive UI to adjust parameters and see the effect on total expenditure.

## Key findings

**Expenditure drivers (EDA)**
- Service-type dataset: total spend is driven most by unit cost (r = 0.79) and application count (r = 0.64).
- Institution dataset: application count (r = 0.79) and per-capita cost (r = 0.73) dominate; public hospitals show the highest average total spend.
- Prescription dataset: average prescription cost (r = 0.80) and prescription count (r = 0.74) dominate; health inflation tracks year almost perfectly (r = 0.99).

**Modeling and the overfitting analysis (the main result)**
- Random Forest reached R² ≈ 0.999 on the service-type dataset — but this was diagnosed as **overfitting**, not success:
  - learning curves showed a persistent train–validation gap,
  - 5-fold cross-validation produced very high error variance,
  - permutation importance revealed the model leaned almost entirely on `Elderly Ratio` and `Year`.
- Retuning model complexity (e.g. `max_depth=10`, `min_samples_leaf=5`) produced a more balanced model; an over-simplified version instead underfit — illustrating the bias–variance trade-off directly.

## What-if scenario module

A Gradio app (launched from notebook `08`) with sliders for visit-count, unit-cost and disease-rate changes (−20% to +20%) and ready-made scenario templates (Baseline, Optimistic, Crisis). It runs the three dataset models on the adjusted inputs and shows base-vs-scenario spending as a table, bar/pie/line charts and a summary sentence — a decision-support aid for proactive, evidence-based budget planning.

## Tech stack

- **Language:** Python
- **ML / data:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Interface:** interactive Gradio app (launched from the what-if notebook)

## Setup

```bash
git clone https://github.com/baharerol/ai-driven-health-expenditure-prediction.git
cd ai-driven-health-expenditure-prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Launch Jupyter and open the notebooks:

```bash
jupyter notebook
```

- For the what-if tool, open `notebooks/08_what_if_analysis.ipynb` and run the cells — this launches an interactive Gradio interface in the browser. The required models are included, so no retraining is needed.
- To reproduce the full pipeline, run the notebooks in order: `01` (data prep) → `02`–`04` (EDA) → `05`–`07` (modeling) → `08` (what-if).

> **Models:** the smaller trained models (including the three the what-if tool loads) are included under `models/`, so the what-if tool runs without retraining. The two largest Random Forest models are not tracked — regenerate them by running the modeling notebooks.

## Limitations

- All data is synthetic, generated to approximate realistic distributions. The models therefore learn the chosen generation process, so the accuracy figures validate the *methodology*, not real-world expenditure — the work is a prototype and decision-support demonstration, not a production forecast.

## Author

Bahar Erol
