
#  Default Prediction — Exploratory V3

**Notebook:** `default_prediction_exploratory_v3.ipynb`  
**Author:** Carllos Watts-Nogueira  
**Date:** August 2025  
**Focus:** Deep Learning, PCA, Stratified K-Fold Validation  
**Tools:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy  

---

##  Badges

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![Model](https://img.shields.io/badge/Model-Sequential%20NN-lightgrey.svg)
![Dimensionality](https://img.shields.io/badge/PCA-180%20components-purple.svg)
![Validation](https://img.shields.io/badge/CV-Stratified%20KFold%20(5)-green.svg)
![Performance](https://img.shields.io/badge/AUC-0.74-success.svg)
![Status](https://img.shields.io/badge/Stage-Exploratory-yellow.svg)

---

##  Project Overview

This version explores a robust deep learning pipeline for predicting loan default risk using TensorFlow and Keras. Key enhancements include:

- Strategic data cleaning  
- Class balancing via undersampling  
- Dimensionality reduction with PCA  
- Stratified K-Fold validation  
- Full pipeline saving for deployment-readiness  

**Dataset Summary:**

- Records: 307,511  
- Original Features: 122  
- Encoded Features: 293  
- Target: `TARGET` (binary: 0 = non-default, 1 = default)

---

##  Data Cleaning & Feature Engineering

### Imputation Strategies

| Category                    | Action                     | Rationale                        |
|----------------------------|----------------------------|----------------------------------|
| Low Missingness (≤1%)      | Median Imputation          | Preserves distribution           |
| Moderate (10–20%)          | Mean Imputation            | Compatible with scaling          |
| Bureau Features            | Zero + Missing Flags       | Behavioral signal                |
| High Missingness (>50%)    | Median + Flags             | Absence may be predictive        |
| Categorical (Light)        | Mode Imputation            | Preserves category distribution  |
| Categorical (Heavy)        | "Unknown" + Flags          | Treats missingness as signal     |
| String Normalization       | `.str.lower().strip()`     | Prevents encoding duplication    |

Additional steps:
- Dropped `SK_ID_CURR` (non-predictive)  
- Created binary flags for missingness  
- Preserved all rows (`dropna()` avoided)

---

##  Class Balancing

Original default rate: **8.07%**  
Balanced using undersampling:

```python
non_default = df[df['TARGET'] == 0]
default = df[df['TARGET'] == 1]

non_default_sample = resample(non_default, replace=False, n_samples=len(default), random_state=42)
balanced_df = pd.concat([non_default_sample, default])
```

- Final shape: `(49,650, 181)`  
- Encoding: `pd.get_dummies(drop_first=True)`  
- Final feature count: **293**

---

##  Dimensionality Reduction with PCA

- Applied after scaling  
- Components tested: 140, 180, 200  
- Best performance: **180 components**  
- PCA improved generalization and reduced overfitting

```python
pca = PCA(n_components=180)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

---

##  Deep Learning Model

```python
model = keras.Sequential([
    layers.Input(shape=(180,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

- Optimizer: Adam (`lr = 0.0005`)  
- Loss: Binary Crossentropy  
- Metrics: Accuracy, AUC  
- EarlyStopping: `monitor='val_auc'`, `patience=3`, `mode='max'`

---

##  Final Evaluation (PCA = 200)

| Metric   | Value |
|----------|-------|
| Accuracy | 0.68  |
| Recall   | 0.69  |
| AUC      | 0.74  |

### Confusion Matrix

```
[[TN FP]
 [FN TP]]
```

### Classification Report

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.68      | 0.67   | 0.68     |
| 1     | 0.69      | 0.69   | 0.69     |

---

##  Stratified K-Fold Validation

- Folds: 5  
- PCA Components Tested: 140, 180, 200  
- Metrics: Accuracy, Recall, AUC, Val_Loss

| PCA | Accuracy | Recall | AUC  | Val_Loss |
|-----|----------|--------|------|----------|
| 140 | 0.67     | 0.64   | 0.73 | 0.61     |
| 180 | 0.68     | 0.69   | 0.74 | 0.60     |
| 200 | 0.68     | 0.67   | 0.74 | 0.60     |

---

##  Model & Pipeline Saving

```python
dump(scaler, 'scaler.pkl')
dump(pca, 'pca_180.pkl')
model_final.save('model_final_180.keras')
```

---

##  Reflections

- PCA improved model stability and interpretability  
- High recall makes this model suitable for risk-sensitive applications  
- Stratified K-Fold validation confirmed generalization across folds  
- Future work: explore SMOTE, ensemble models, and deployment pipelines

---

##  Comparison: V2 vs V3

| Feature              | V2                          | V3                          | What's New in V3                     |
|----------------------|-----------------------------|-----------------------------|--------------------------------------|
| Data Cleaning        | Strategic imputations       | Same                        | No change                            |
| Row Preservation     | No rows dropped             | Same                        | Both avoid `dropna()`                |
| Encoding             | `get_dummies(drop_first)`   | Same                        | Identical encoding                   |
| Class Balancing      | Undersampling               | Same                        | Same technique                       |
| Model Architecture   | 64 → Dropout → 32 → Sigmoid | 128 → Dropout → 64 → 32 → Sigmoid | Deeper and more expressive         |
| PCA                  | Not used                    | Applied (180 components)    | Dimensionality reduction added       |
| Input Data           | Scaled                      | PCA-transformed             | More compact and informative input   |
| EarlyStopping        | `val_auc`                   | `val_auc` + `mode='max'`    | Slight tuning improvement            |
| Evaluation           | Accuracy, Recall, AUC       | Same + ROC Curve            | ROC visualization added              |
| Cross-Validation     | Not implemented             | Stratified K-Fold (5 folds) | Robustness testing added             |
| Pipeline Saving      | Not specified               | Full pipeline saved         | Ready for reuse and deployment       |
| Final Report         | Metrics and reflections     | Detailed + plots            | More complete and visual reporting   |

---

##  Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---
