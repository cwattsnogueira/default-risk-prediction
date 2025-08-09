Default Prediction Project — Final README

Overview

This project aims to predict customer loan default risk using a combination of deep learning and ensemble methods. It evolved through multiple exploratory stages, culminating in a production-ready pipeline that integrates best practices in data preprocessing, model training, evaluation, and artifact saving.

The project is organized into four main phases:
- Exploratory V2 — Baseline deep learning pipeline with strategic cleaning and undersampling.
- Exploratory V3 — Enhanced version with PCA, cross-validation, and model saving.
- Pre-Final — Course project with five modeling strategies and extensive evaluation.
- Final Version — Consolidated and optimized models ready for deployment.

Project Structure

```
default_prediction_project/
├── final_version_default_prediction/
│   ├── models/
│   ├── notebooks/
│   ├── data/
│   ├── README.md
│   └── requirements.txt
├── default_prediction_exploratory_pre_final/
├── default_prediction_exploratory_v2/
├── default_prediction_exploratory_v3/
└── README.md
```


Each subfolder contains its own notebook, scripts, and documentation.

Phase Summaries

1. Exploratory V2
- Goal: Build a clean, reproducible deep learning pipeline.
- Highlights:
- Strategic imputation based on missingness severity.
- Undersampling to balance classes.
- One-hot encoding and scaling.
- Simple neural network: 64 → Dropout → 32 → Sigmoid.
- Evaluation: Accuracy (0.68), Recall (0.67), AUC (0.74).
- Limitations: No dimensionality reduction, cross-validation, or model saving.

2. Exploratory V3
- Goal: Improve generalization and modularity.
- Enhancements:
- PCA applied (tested 140, 180, 200 components).
- Stratified K-Fold validation (5 folds).
- Deeper architecture: 128 → Dropout → 64 → 32 → Sigmoid.
- ROC curve visualization added.
- Saved artifacts: scaler.pkl, pca_180.pkl, model_final_180.keras.
- Best PCA config: 180 components yielded highest recall and AUC.

3. Pre-Final (Course Project)
- Goal: Explore multiple modeling strategies.
- Models:
- MLP with undersampling, class weights, SMOTE.
- PCA + K-Fold + Class Weights.
- LightGBM with SMOTE and threshold tuning.
- Key Learnings:
- Data quality > quantity.
- Threshold tuning significantly affects performance.
- LightGBM outperformed deep models on tabular data.
- Best Model: Model 4 (LightGBM + SMOTE) with AUC 0.992, F1 0.971.

4. Final Version
- Goal: Deliver a clean, deployable solution.
- Saved Models: 
---

| Model   | Strategy               | File                             |
|---------|------------------------|----------------------------------|
| model_1 | Undersampling + NN     | `model_1_undersampling_nn.keras` |
| model_2 | Class Weights + MLP    | `model_2_class_weights_mlp.keras` |
| model_3 | PCA + K-Fold NN        | `model_3_pca_kfold_nn.keras`     |
| model_4 | LightGBM               | `model_4_lightgbm.pkl`           |
| model_5 | SMOTE + NN             | `model_5_smote_nn.keras`         |

---

- Performance Summary:

| Model | AUC  | F1   | Precision | Recall | Accuracy |
|-------|------|------|-----------|--------|----------|
| M1    | 0.78 | 0.65 | 0.68      | 0.62   | 0.74     |
| M2    | 0.81 | 0.69 | 0.72      | 0.66   | 0.76     |
| M3    | 0.83 | 0.71 | 0.74      | 0.68   | 0.78     |
| M4    | 0.86 | 0.75 | 0.78      | 0.72   | 0.81     |
| M5    | 0.80 | 0.68 | 0.70      | 0.66   | 0.75     |

- Recommendation:
- Use model_4_lightgbm.pkl for best performance.
- Use model_3_pca_kfold_nn.keras if dimensionality reduction is required.

How to Use the Saved Models

1. Load Scaler
import pickle
with open('models/scaler_model1.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_new_scaled = scaler.transform(X_new)


2. Load Model
from keras.models import load_model
model = load_model('models/model_1_undersampling_nn.keras')
y_pred = model.predict(X_new_scaled)

Requirements
Install dependencies:
pip install -r requirements.txt

Lessons Learned
- Strategic imputation and flag creation improve model interpretability.
- PCA enhances generalization and reduces overfitting.
- Cross-validation is essential for robust performance estimation.
- Threshold tuning can dramatically shift model behavior.
- LightGBM excels on tabular data, but neural nets offer flexibility.

Production Readiness

| Model | Suitability      |
|-------|------------------|
| M1    | Prototype        |
| M2    | Needs tuning     |
| M3    | Experimental     |
| M4    | Production-ready |
| M5    | Production-ready |



All models are saved with corresponding scalers and PCA (if applicable), making them easy to deploy in APIs or dashboards.

---

## Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---



