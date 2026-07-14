# Car Price Prediction — RF vs XGBoost

> Regression benchmark on ~10,000 used-car listings comparing **Random Forest** and **XGBoost** for predicting sale price.
>
> **SSIM916 — Machine Learning for Social Data Science, University of Exeter**

---

## 🎯 Problem

Accurate car pricing helps buyers, dealerships, insurers and lenders. This project builds and compares two ensemble regressors on a tabular listings dataset and analyses which features drive price most.

## 📦 Dataset

- **Rows:** ~10,000 car listings
- **Features (11):** `Brand`, `Model`, `Year`, `Engine Capacity`, `Fuel Type`, `Transmission`, `Mileage`, `Doors`, `Owners`
- **Target:** `Price`
- **Split:** 80 / 20 train / test (`random_state=42`)

## 🧪 Methodology

1. Label-encode categorical features (`Brand`, `Model`, `Fuel_Type`, `Transmission`)
2. Handle nulls, normalise numeric columns
3. Train two models with default-ish hyperparameters (`n_estimators=100`)
4. Evaluate on held-out test set with **MSE** and **R²**
5. Inspect feature importance from XGBoost

## 📈 Results

| Model | MSE ↓ | R² ↑ |
|---|---:|---:|
| **XGBoost** | **65,564.60** | **0.99** |
| Random Forest | 239,412.52 | 0.97 |

> ⚠️ **Caveat:** the R² of ~0.99 is unusually high and likely reflects synthetic / near-clean data. In interviews I frame this as a *methodology* project (pipeline, benchmark, feature importance) rather than a claim about real-world generalisation.

**Why XGBoost wins here:** sequential boosting corrects prior residuals, L1+L2 regularisation controls overfitting, and the optimised histogram-based tree builder handles the tabular structure efficiently.

## 🧰 Tech Stack

`Python` · `scikit-learn` · `xgboost` · `pandas` · `numpy` · `matplotlib`

## 📁 Repo Structure

```text
.
├── notebooks/car_price_prediction.ipynb
├── src/
├── docs/Car_Price_Prediction_Report.pdf
├── requirements.txt
└── README.md
```

## ▶️ Reproduce

```bash
pip install -r requirements.txt
jupyter lab notebooks/car_price_prediction.ipynb
```

## 📄 Report

Full write-up → [`docs/Car_Price_Prediction_Report.pdf`](docs/Car_Price_Prediction_Report.pdf)

---

<sub>MIT-licensed · Author: [Parth Badiger](https://github.com/parthbadiger24-creator)</sub>
