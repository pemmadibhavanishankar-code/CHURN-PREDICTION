# Customer Churn Prediction: A Comparative Machine Learning Approach

An end-to-end Python framework utilizing **Scikit-Learn** pipelines to build, validate, and compare supervised classification models for customer churn prediction.

---

## 📌 Project Overview

This repository contains a modular machine learning pipeline designed to predict customer churn (`binary classification`). The project demonstrates robust data engineering habits—specifically using **Scikit-Learn Pipelines** and **ColumnTransformers** to eliminate data leakage—and compares a classic linear baseline against a non-linear ensemble tree model.

### Key Highlights
* **Automated Preprocessing:** Seamless handling of heterogeneous data types (continuous numerical scaling and categorical text encoding).
* **Robust Validation:** Implements 5-Fold Stratified Cross-Validation to assess generalized performance boundaries on imbalanced targets.
* **Leakage Prevention:** Bundles preprocessing operations directly into model execution pipelines to guarantee zero data bleeding across folds.

---

## 🛠️ Tech Stack & Dependencies

The system relies entirely on standard data science and machine learning libraries:
* **Core Data Engine:** `numpy`, `pandas`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`

To install the dependencies, execute
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
