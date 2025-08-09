# Default Prediction — Final Version

This project aims to predict customer default risk based on financial and demographic data. The `final_version_default_prediction` notebook consolidates best practices in modeling, preprocessing, and artifact saving for production-ready reuse.

## Objective

Predict the likelihood of customer default using supervised learning techniques, with a focus on:

- Class imbalance handling (undersampling, SMOTE, class weights)
- Dimensionality reduction (PCA)
- Neural networks and tree-based models (LightGBM)

## Trained Models

| Model      | Strategy                     | File                                 |
|------------|------------------------------|--------------------------------------|
| `model_1`  | Undersampling + Neural Net   | `model_1_undersampling_nn.keras`     |
| `model_2`  | Class Weights + MLP          | `model_2_class_weights_mlp.keras`    |
| `model_3`  | PCA + K-Fold Neural Net      | `model_3_pca_kfold_nn.keras`         |
| `model_4`  | LightGBM                     | `model_4_lightgbm.pkl`               |
| `model_5`  | SMOTE + Neural Net           | `model_5_smote_nn.keras`             |


## Model Performance Summary

| Model ID | Strategy                     | ROC AUC | F1-Score | Precision | Recall | Accuracy |
|----------|------------------------------|--------:|---------:|----------:|-------:|---------:|
| `model_1` | Undersampling + Neural Net   | 0.78    | 0.65     | 0.68      | 0.62   | 0.74     |
| `model_2` | Class Weights + MLP          | 0.81    | 0.69     | 0.72      | 0.66   | 0.76     |
| `model_3` | PCA + K-Fold Neural Net      | 0.83    | 0.71     | 0.74      | 0.68   | 0.78     |
| `model_4` | LightGBM                     | 0.86    | 0.75     | 0.78      | 0.72   | 0.81     |
| `model_5` | SMOTE + Neural Net           | 0.80    | 0.68     | 0.70      | 0.66   | 0.75     |

> Note: All metrics are based on the test set. LightGBM (`model_4`) achieved the best overall performance.

## Recommendation

For deployment or API integration, we recommend using:

- `model_4_lightgbm.pkl` for best performance
- `model_3_pca_kfold_nn.keras` if dimensionality reduction is required

## How to Use the Saved Models

To make predictions on new data:

### 1. Load the corresponding scaler

```python
import pickle
with open('models/scaler_model1.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_new_scaled = scaler.transform(X_new)

### 2. Load the model

from keras.models import load_model
model = load_model('models/model_1_undersampling_nn.keras')
y_pred = model.predict(X_new_scaled)

Project Structure

final_version_default_prediction/
├── models/
│   ├── model_1_undersampling_nn.keras
│   ├── model_2_class_weights_mlp.keras
│   ├── model_3_pca_kfold_nn.keras
│   ├── model_4_lightgbm.pkl
│   ├── model_5_smote_nn.keras
│   ├── scaler_model1.pkl
│   ├── scaler_model2.pkl
│   ├── scaler_model3.pkl
│   ├── scaler_model4.pkl
│   ├── scaler_model5.pkl
│   └── pca_150.pkl
├── notebooks/
│   └── final_version_default_prediction.ipynb and .py
├── data/
├── README.md
└── requirements.txt

Requirements
Install dependencies with:
pip install -r requirements.txt

Notes
- Each model was trained using a distinct strategy for class balancing or dimensionality reduction.
- Scalers were saved separately to ensure consistent preprocessing during inference.
- PCA was fitted with 150 components and can be optionally applied.
- All models are ready for deployment or integration into APIs and dashboards.

Contact
Author: Carllos Watts-Nogueira
Email: [carlloswattsnogueira@gmail.com]
LinkedIn: [https://www.linkedin.com/in/carlloswattsnogueira/]
