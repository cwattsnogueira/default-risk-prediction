# Deep Learning for Default Prediction — Course End Project

This project explores multiple strategies for predicting customer default using deep learning and ensemble methods. It was developed as part of the "Deep Learning with TensorFlow and Keras" course and includes five distinct modeling approaches, each designed to address challenges such as class imbalance, feature dimensionality, and generalization.

Author: Carllos Watts-Nogueira  
Date: July 12  
Notebook: `default_prediction_exploratory_pre_final_models.ipynb`

## Project Objective

To build and evaluate models that predict whether a customer will default on a loan, using a dataset with financial and demographic features. The project focuses on:

- Handling missing data  
- Addressing class imbalance  
- Feature encoding and scaling  
- Building neural networks and LightGBM models  
- Evaluating performance across multiple thresholds  
- Comparing undersampling, SMOTE, and class weighting strategies  

## Modeling Overview

| Model ID | Strategy                                | Algorithm      | Balancing Method | Threshold | Notes |
|----------|-----------------------------------------|----------------|------------------|-----------|-------|
| Model 1  | Undersampling + MLP                     | Neural Network | Undersampling    | 0.5       | Small balanced dataset |
| Model 2  | Class Weights + MLP                     | Neural Network | Class weights    | 0.5 / 0.3 | Full dataset |
| Model 3  | PCA + K-Fold + Class Weights            | Neural Network | Class weights    | 0.3       | Dimensionality reduction |
| Model 4  | SMOTE + LightGBM + Threshold Tuning     | LightGBM       | SMOTE            | Tuned     | Optimized for F1 |
| Model 5  | SMOTE + MLP                             | Neural Network | SMOTE            | 0.5       | High-performing deep model |


## Step-by-Step Workflow

### 1. Data Loading & Cleaning

- Loaded `loan_data.csv` and `Data_Dictionary.csv`.
- Identified columns with missing values and categorized them by severity.
- Dropped all rows with missing values, reducing the dataset from 307,511 to 8,602 rows.
- Removed placeholder values like `'None'` and `'Missing'` from object columns.
- Dropped duplicate rows and irrelevant ID column (`SK_ID_CURR`).

### 2. Exploratory Data Analysis (EDA)

- Visualized distributions of key numeric features (`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`).
- Used heatmaps to inspect feature correlations.
- Analyzed how categorical variables (e.g., `OCCUPATION_TYPE`) relate to default rates.
- Found that only 6.11% of customers had `TARGET = 1` (default), confirming class imbalance.

### 3. Class Balancing Strategies

- **Model 1**: Used undersampling to balance the dataset (1,052 rows total).
- **Model 2 & 3**: Used `class_weight='balanced'` during training.
- **Model 4 & 5**: Applied SMOTE to synthesize minority class examples.

### 4. Feature Engineering

- One-hot encoded categorical variables using `pd.get_dummies()`.
- Scaled features using `StandardScaler`.
- Applied PCA (Model 3) to reduce dimensionality to 150 components.

### 5. Model Architectures

#### Neural Networks (Models 1, 2, 3, 5)

- Input → Dense(128) → BatchNorm → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Output(sigmoid)
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metrics: Accuracy, AUC  
- EarlyStopping used to prevent overfitting

#### LightGBM (Model 4)

- Tuned hyperparameters: learning rate, depth, leaves, subsample, colsample  
- Used `class_weight='balanced'`  
- Threshold optimized for F1-score  
- Feature importance plotted  

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-score | AUC   | Threshold |
|-------|----------|-----------|--------|----------|-------|-----------|
| M1    | 0.65     | 0.62      | 0.69   | 0.65     | 0.68  | 0.5       |
| M2    | 0.795    | 0.121     | 0.436  | 0.188    | 0.695 | 0.5       |
| M2*   | 0.68     | 0.105     | 0.649  | 0.18     | 0.695 | 0.3       |
| M3    | 0.684    | 0.099     | 0.515  | 0.165    | 0.664 | 0.3       |
| M4    | 0.9718   | 0.9954    | 0.9480 | 0.9711   | 0.992 | 0.42      |
| M5    | 0.97     | 0.99      | 0.94   | 0.96     | 0.981 | 0.5       |

> Note: Model 4 (LightGBM + SMOTE + threshold tuning) achieved the best overall performance, with near-perfect precision and recall.

## Visualizations

- ROC and Precision-Recall curves plotted for Models 4 and 5.
- PCA explained variance curve plotted to justify component selection.
- Confusion matrices visualized for all models.

## Lessons Learned

- **Data quality > data quantity**: Dropping rows with excessive nulls improved model reliability.
- **Balancing matters**: Undersampling, SMOTE, and class weights each affect recall and precision differently.
- **Threshold tuning is powerful**: Adjusting decision boundaries can dramatically shift model behavior.
- **Neural networks vs. LightGBM**: LightGBM outperformed deep models on tabular data, but neural nets offer flexibility.
- **Evaluation beyond accuracy**: Precision, recall, and AUC are more informative in imbalanced classification.

## Production Readiness

| Model | Pros | Cons | Suitability |
|-------|------|------|-------------|
| M1    | Balanced recall/precision | Small dataset | Prototype |
| M2    | Full data usage | Low precision | Needs tuning |
| M3    | Generalization via K-Fold | Low precision | Experimental |
| M4    | High precision & recall | Complex pipeline | Production-ready |
| M5    | Strong deep model | Slightly lower AUC than M4 | Production-ready |

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

Diference between final_version_default_prediction.ipynb and default_prediction_exploratory_pre_final_models.ipynb

Key Differences: Exploratory vs Final Project
| Aspect | Course Project (default_prediction_exploratory_pre_final_models.ipynb) | Final Project (final_version_default_prediction.ipynb) | 
| Data Cleaning | Dropped all rows with missing values | Likely used smarter imputation or selective column removal | 
| Balancing Strategy | Tested undersampling, class weights, SMOTE | Finalized on SMOTE and class weights based on performance | 
| Model Variety | 5 models: MLPs, PCA, LightGBM, SMOTE | Selected best-performing models for saving and deployment | 
| Threshold Tuning | Explicit threshold optimization (e.g., 0.3, 0.42) | Final models likely use tuned thresholds or default 0.5 | 
| Evaluation Depth | Extensive metric tracking (precision, recall, AUC, F1) | Final report summarizes metrics and recommends best model | 
| Cross-Validation | Used Stratified K-Fold in Model 3 | Final version may rely on holdout or best fold strategy | 
| Feature Engineering | One-hot encoding, PCA, scaling | Final version includes saved scalers and PCA for reuse | 
| Model Saving | No model saving in this version | Final version saves .keras, .pkl, and .pca artifacts | 
| Production Readiness | Mostly experimental and educational | Final version is structured for deployment and reuse | 
| Documentation | Embedded in notebook comments and markdown | Final version includes structured README with usage instructions | 

Strategic Evolution

What I Refined in the Final Version:
- Cleaner pipeline: Removed exploratory clutter, focused on reproducibility.
- Artifact saving: Models, scalers, PCA saved for future use.
- Modular structure: Clear separation of training, evaluation, and deployment logic.
- Client-ready documentation: README with model usage, metrics, and structure.
- Performance focus: Chose best model based on business-relevant metrics (e.g., recall for risk detection).

Summary
The course project was my sandbox — where I tested ideas, architectures, and preprocessing strategies.
The final version is my polished deliverable — optimized, reproducible, and ready for stakeholders or deployment.

Contact
Author: Carllos Watts-Nogueira
Email: [carlloswattsnogueira@gmail.com]
LinkedIn: [https://www.linkedin.com/in/carlloswattsnogueira/]