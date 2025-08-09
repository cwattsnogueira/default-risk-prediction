# Default Prediction with Deep Learning - default_prediction_exploratory_v2.ipynb

## Project Overview

This project aims to predict loan default risk using a deep learning model built with TensorFlow and Keras. The dataset contains over 300,000 records and 122 variables related to customer demographics, financial behavior, and credit history. The workflow emphasizes **data integrity**, **strategic imputation**, and **class balancing** to ensure robust model performance.

## Project Structure
default_prediction_project/
├── final_version_default_prediction/
│   ├── models/
│   ├── notebooks/
│   ├── data/
│   ├── README.md
│   └── requirements.txt
├── default_prediction_exploratory_pre_final/
│   ├── default_prediction_exploratory_pre_final_models.ipynb
│   ├── default_prediction_exploratory_pre_final_models.py
│   └── README.md
├── default_prediction_exploratory_v2/
│   ├── default_prediction_exploratory_v2.ipynb
│   ├── default_prediction_exploratory_v2.py
│   └── README.md
├── default_prediction_exploratory_v3/
│   ├── default_prediction_exploratory_v3.ipynb
│   ├── default_prediction_exploratory_v3.py
│   └── README.md
└── README.md  

## Data Cleaning & Feature Engineering

### Strategies Applied

| Category | Strategy | Rationale |
|---------|----------|-----------|
| Low Missingness (≤1%) | Median Imputation | Preserves distribution |
| Moderate Missingness (10–20%) | Mean Imputation | Features are normalized |
| Bureau Features | Impute with 0 + Missing Flags | Missingness may reflect behavior |
| High Missingness (>50%) | Median Imputation + Flags | Absence may be predictive |
| Categorical (Light Missingness) | Mode Imputation | Preserves category distribution |
| Categorical (Heavy Missingness) | "Unknown" + Flags | Treats missingness as signal |
| String Normalization | `.str.lower().strip()` | Prevents encoding duplication |

Additional steps:
- Dropped `SK_ID_CURR` (non-predictive)
- Created binary flags for missingness
- Preserved all rows (no `dropna()`)

## Class Balancing

The dataset was highly imbalanced (only 8.07% defaults). To address this:

```python
from sklearn.utils import resample

non_default = df[df['TARGET'] == 0]
default = df[df['TARGET'] == 1]

non_default_sample = resample(non_default, replace=False, n_samples=len(default), random_state=42)
balanced_df = pd.concat([non_default_sample, default])

- Final balanced shape: (49,650, 181)
- Encoding via pd.get_dummies(drop_first=True)
- Final feature set: 293 columns

Model Architecture

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy, AUC
- EarlyStopping on val_auc

Evaluation Metrics
| Metric | Value | 
| Accuracy | 0.68 | 
| Recall (Sensitivity) | 0.67 | 
| ROC AUC | 0.74 | 

Confusion Matrix
[[3422 1531]
 [1636 3341]]

Classification Report
| Class | Precision | Recall | F1-score | Support | 
| 0 | 0.68 | 0.69 | 0.68 | 4953 | 
| 1 | 0.69 | 0.67 | 0.68 | 4977 | 

Reflections & Open Questions
- Is selective imputation + flag creation the most robust strategy?
- Could alternative balancing techniques (e.g. SMOTE) yield better generalization?
- How replicable is this pipeline across different datasets?
- What improvements could be made for deployment-readiness?


Contact
Author: Carllos Watts-Nogueira
Email: [carlloswattsnogueira@gmail.com]
LinkedIn: [https://www.linkedin.com/in/carlloswattsnogueira/]

