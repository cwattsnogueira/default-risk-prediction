#  Default Prediction — Final Version

**Notebook:** `final_version_default_prediction.ipynb`  
**Author:** Carllos Watts-Nogueira  
**Date:** August 2025  
**Focus:** Production-ready ML pipeline for default risk prediction  
**Tools:** TensorFlow, Keras, LightGBM, Scikit-learn, PCA  

---

##  Badges

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-green.svg)
![Model Count](https://img.shields.io/badge/Models-5%20trained-lightgrey.svg)
![Dimensionality](https://img.shields.io/badge/PCA-150%20components-purple.svg)
![Deployment](https://img.shields.io/badge/Ready%20for-API%20%2F%20Dashboard-success.svg)

---

##  Objective

This project predicts customer default risk using supervised learning techniques. The final version consolidates best practices in:

- Class imbalance handling: undersampling, SMOTE, class weights  
- Dimensionality reduction: PCA (150 components)  
- Model diversity: neural networks and tree-based models  
- Artifact saving: scalers, PCA, and models for reuse  

---

##  Trained Models

| Model ID   | Strategy                     | File Name                        |
|------------|------------------------------|----------------------------------|
| `model_1`  | Undersampling + Neural Net   | `model_1_undersampling_nn.keras` |
| `model_2`  | Class Weights + MLP          | `model_2_class_weights_mlp.keras`|
| `model_3`  | PCA + K-Fold Neural Net      | `model_3_pca_kfold_nn.keras`     |
| `model_4`  | LightGBM                     | `model_4_lightgbm.pkl`           |
| `model_5`  | SMOTE + Neural Net           | `model_5_smote_nn.keras`         |

---

##  Model Performance Summary

| Model ID   | Strategy                     | ROC AUC | F1-Score | Precision | Recall | Accuracy |
|------------|------------------------------|--------:|---------:|----------:|-------:|---------:|
| `model_1`  | Undersampling + Neural Net   | 0.78    | 0.65     | 0.68      | 0.62   | 0.74     |
| `model_2`  | Class Weights + MLP          | 0.81    | 0.69     | 0.72      | 0.66   | 0.76     |
| `model_3`  | PCA + K-Fold Neural Net      | 0.83    | 0.71     | 0.74      | 0.68   | 0.78     |
| `model_4`  | LightGBM                     | 0.86    | 0.75     | 0.78      | 0.72   | 0.81     |
| `model_5`  | SMOTE + Neural Net           | 0.80    | 0.68     | 0.70      | 0.66   | 0.75     |

>  LightGBM (`model_4`) achieved the best overall performance on the test set.

---

##  Recommendation

For deployment or API integration:

- Use `model_4_lightgbm.pkl` for highest performance  
- Use `model_3_pca_kfold_nn.keras` if dimensionality reduction is required  

---

##  How to Use the Saved Models

### 1. Load the Scaler

```python
import pickle
with open('models/scaler_model1.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_new_scaled = scaler.transform(X_new)
```

### 2. Load the Model

```python
from keras.models import load_model
model = load_model('models/model_1_undersampling_nn.keras')
y_pred = model.predict(X_new_scaled)
```

---

##  Project Structure

```
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
│   └── final_version_default_prediction.ipynb / .py
├── data/
├── README.md
└── requirements.txt
```

---

##  Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

##  Notes

- Each model uses a distinct strategy for class balancing or dimensionality reduction  
- Scalers are saved separately to ensure consistent preprocessing  
- PCA was fitted with 150 components and is optional  
- All models are ready for deployment or integration into APIs and dashboards  

---

##  Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---

