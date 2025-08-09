# Deep Learning for Default Prediction — Course End Project

**Author:** Carllos Watts-Nogueira  
**Date:** July 12, 2025  
**Notebook:** `default_prediction_exploratory_pre_final_models.ipynb`  
**Course:** Deep Learning with TensorFlow and Keras  
**Institution:** University of San Diego / Fullstack Academy  

---

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)  
![Best Model](https://img.shields.io/badge/Best%20Model-LightGBM%20(AUC%3D0.992)-success.svg)  
![Class Imbalance](https://img.shields.io/badge/Class%20Imbalance-SMOTE%20%26%20Weights-orange.svg)  
![Dimensionality](https://img.shields.io/badge/Dimensionality-PCA%20(150)-yellow.svg)  
![Evaluation](https://img.shields.io/badge/Evaluation-AUC%2C%20F1%2C%20Recall%2C%20Precision-lightgrey.svg)

---

##  Project Objective

To build and evaluate models that predict customer loan default using financial and demographic data. The project addresses:

- Missing data handling  
- Class imbalance mitigation  
- Feature encoding and scaling  
- Deep learning and ensemble modeling  
- Threshold tuning and performance comparison  

---

##  Modeling Strategies

| Model ID | Strategy                            | Algorithm      | Balancing Method | Threshold | Notes                          |
|----------|-------------------------------------|----------------|------------------|-----------|--------------------------------|
| M1       | Undersampling + MLP                 | Neural Network | Undersampling    | 0.5       | Small balanced dataset         |
| M2       | Class Weights + MLP                 | Neural Network | Class weights    | 0.5 / 0.3 | Full dataset                   |
| M3       | PCA + K-Fold + Class Weights        | Neural Network | Class weights    | 0.3       | Dimensionality reduction       |
| M4       | SMOTE + LightGBM + Threshold Tuning | LightGBM       | SMOTE            | 0.42      | Optimized for F1-score         |
| M5       | SMOTE + MLP                         | Neural Network | SMOTE            | 0.5       | High-performing deep model     |

---

##  Workflow Summary

### 1. Data Cleaning  
- Dropped rows with missing values (from 307,511 to 8,602)  
- Removed placeholders (`'None'`, `'Missing'`)  
- Dropped duplicates and irrelevant ID column  

### 2. EDA  
- Distribution plots for income, credit, annuity  
- Correlation heatmaps  
- Categorical analysis (e.g., `OCCUPATION_TYPE`)  
- Confirmed class imbalance: 6.11% defaults  

### 3. Balancing Techniques  
- Undersampling (Model 1)  
- Class weights (Models 2 & 3)  
- SMOTE (Models 4 & 5)  

### 4. Feature Engineering  
- One-hot encoding  
- StandardScaler  
- PCA (Model 3: 150 components)  

### 5. Model Architectures  
**Neural Networks (M1, M2, M3, M5):**  
- Input → Dense(128) → BatchNorm → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Output(sigmoid)  
- Optimizer: Adam | Loss: Binary Crossentropy | Metrics: Accuracy, AUC  
- EarlyStopping applied  

**LightGBM (M4):**  
- Tuned hyperparameters  
- Class weights enabled  
- Threshold optimized for F1  
- Feature importance visualized  

---

##  Model Performance

| Model | Accuracy | Precision | Recall | F1-score | AUC   | Threshold |
|-------|----------|-----------|--------|----------|-------|-----------|
| M1    | 0.65     | 0.62      | 0.69   | 0.65     | 0.68  | 0.5       |
| M2    | 0.795    | 0.121     | 0.436  | 0.188    | 0.695 | 0.5       |
| M2*   | 0.68     | 0.105     | 0.649  | 0.18     | 0.695 | 0.3       |
| M3    | 0.684    | 0.099     | 0.515  | 0.165    | 0.664 | 0.3       |
| M4    | 0.9718   | 0.9954    | 0.9480 | 0.9711   | 0.992 | 0.42      |
| M5    | 0.97     | 0.99      | 0.94   | 0.96     | 0.981 | 0.5       |

> **Model 4** (LightGBM + SMOTE + threshold tuning) achieved the best overall performance.

---

##  Visualizations

- ROC and Precision-Recall curves (Models 4 & 5)  
- PCA explained variance curve  
- Confusion matrices for all models  

---

##  Lessons Learned

- Data quality is more impactful than quantity  
- Balancing methods significantly affect recall and precision  
- Threshold tuning is a powerful lever for classification behavior  
- LightGBM outperforms deep models on tabular data  
- AUC, precision, and recall are more informative than accuracy in imbalanced settings  

---

##  Production Readiness

| Model | Pros                      | Cons                  | Suitability       |
|-------|---------------------------|------------------------|-------------------|
| M1    | Balanced recall/precision | Small dataset          | Prototype         |
| M2    | Full data usage           | Low precision          | Needs tuning      |
| M3    | Generalization via K-Fold | Low precision          | Experimental      |
| M4    | High precision & recall   | Complex pipeline       | Production-ready  |
| M5    | Strong deep model         | Slightly lower AUC     | Production-ready  |

---

##  Project Structure

```
default_risk_prediction/
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
```

---

##  Key Differences: Exploratory vs Final Version

| Aspect             | Exploratory Project                            | Final Project                                  |
|--------------------|------------------------------------------------|------------------------------------------------|
| Data Cleaning      | Dropped all nulls                              | Selective imputation or column removal         |
| Balancing Strategy | Tested multiple methods                        | Finalized on SMOTE + class weights             |
| Model Variety      | 5 models with varied strategies                | Selected best-performing models                |
| Threshold Tuning   | Explicit tuning (e.g., 0.3, 0.42)              | Final models use tuned or default thresholds   |
| Evaluation Depth   | Extensive metric tracking                      | Summarized metrics for deployment              |
| Cross-Validation   | Stratified K-Fold (Model 3)                    | Holdout or best fold strategy                  |
| Feature Engineering| One-hot, PCA, scaling                          | Saved scalers and PCA for reuse                |
| Model Saving       | No saving                                      | Saved `.keras`, `.pkl`, `.pca` artifacts       |
| Documentation      | Markdown and comments in notebook              | Structured README with usage instructions      |

---

##  Strategic Evolution

**Refinements in Final Version:**

- Cleaner pipeline with reproducible logic  
- Saved artifacts for deployment  
- Modular structure separating training and evaluation  
- Business-focused metrics for model selection  
- Client-ready documentation  

---

##  Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---
