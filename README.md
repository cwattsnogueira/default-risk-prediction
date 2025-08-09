# Default Prediction Project

**Submission Date:** May 2025  
**Author:** Carllos Watts-Nogueira  
**Course:** Artificial Intelligence & Machine Learning  
**Institution:** University of San Diego / Fullstack Academy  
**Section:** 2504-FTB-CT-AIM-PT  

---

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)  
![Model](https://img.shields.io/badge/Best%20Model-LightGBM%20(AUC%3D0.86)-success.svg)  
![Pipeline](https://img.shields.io/badge/Pipeline-4%20Phases-yellow.svg)  
![Evaluation](https://img.shields.io/badge/Evaluation-AUC%2C%20F1%2C%20Recall%2C%20Precision-orange.svg)  
![Artifacts](https://img.shields.io/badge/Artifacts-Saved%20Models%20%26%20Scalers-lightgrey.svg)

---

## Overview

This project predicts customer loan default risk using deep learning and ensemble methods. It evolved through four structured phases, culminating in a production-ready pipeline that integrates best practices in preprocessing, modeling, evaluation, and artifact saving.

### Project Phases

1. **Exploratory V2**  
   - Baseline deep learning pipeline  
   - Strategic imputation, undersampling, one-hot encoding  
   - Neural network: 64 → Dropout → 32 → Sigmoid  
   - Metrics: Accuracy 0.68, Recall 0.67, AUC 0.74  

2. **Exploratory V3**  
   - PCA (140–200 components), Stratified K-Fold  
   - Architecture: 128 → Dropout → 64 → 32 → Sigmoid  
   - ROC curve visualization  
   - Saved artifacts: scaler, PCA, model  

3. **Pre-Final (Course Project)**  
   - Five modeling strategies: MLP, PCA, SMOTE, LightGBM  
   - Key insight: LightGBM outperforms deep models on tabular data  
   - Best model: LightGBM + SMOTE (AUC 0.992, F1 0.971)  

4. **Final Version**  
   - Consolidated models ready for deployment  
   - Saved models and scalers for API/dashboard integration  

---

## Project Structure

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

---

## Final Models

| Model   | Strategy               | File                             |
|---------|------------------------|----------------------------------|
| model_1 | Undersampling + NN     | `model_1_undersampling_nn.keras` |
| model_2 | Class Weights + MLP    | `model_2_class_weights_mlp.keras` |
| model_3 | PCA + K-Fold NN        | `model_3_pca_kfold_nn.keras`     |
| model_4 | LightGBM               | `model_4_lightgbm.pkl`           |
| model_5 | SMOTE + NN             | `model_5_smote_nn.keras`         |

---

## Performance Summary

| Model | AUC  | F1   | Precision | Recall | Accuracy |
|-------|------|------|-----------|--------|----------|
| M1    | 0.78 | 0.65 | 0.68      | 0.62   | 0.74     |
| M2    | 0.81 | 0.69 | 0.72      | 0.66   | 0.76     |
| M3    | 0.83 | 0.71 | 0.74      | 0.68   | 0.78     |
| M4    | 0.86 | 0.75 | 0.78      | 0.72   | 0.81     |
| M5    | 0.80 | 0.68 | 0.70      | 0.66   | 0.75     |

**Recommendation:**  
- Use `model_4_lightgbm.pkl` for best performance.  
- Use `model_3_pca_kfold_nn.keras` if dimensionality reduction is required.

---

## How to Use the Saved Models

**1. Load Scaler**
```python
import pickle
with open('models/scaler_model1.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_new_scaled = scaler.transform(X_new)
```

**2. Load Model**
```python
from keras.models import load_model
model = load_model('models/model_1_undersampling_nn.keras')
y_pred = model.predict(X_new_scaled)
```

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Lessons Learned

- Strategic imputation and flag creation improve interpretability  
- PCA enhances generalization and reduces overfitting  
- Cross-validation is essential for robust performance  
- Threshold tuning can dramatically shift model behavior  
- LightGBM excels on tabular data; neural nets offer flexibility  

---

## Production Readiness

| Model | Suitability      |
|-------|------------------|
| M1    | Prototype        |
| M2    | Needs tuning     |
| M3    | Experimental     |
| M4    | Production-ready |
| M5    | Production-ready |

All models are saved with corresponding scalers and PCA (if applicable), ready for deployment in APIs or dashboards.

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

## Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---

