# Default Prediction — Exploratory V3

## Project Overview

This version explores a robust deep learning pipeline for predicting loan default risk using TensorFlow and Keras. It emphasizes **strategic data cleaning**, **class balancing**, **dimensionality reduction via PCA**, and **model tuning with Stratified K-Fold validation**.

Dataset:  
- Records: 307,511  
- Features: 122 original → 293 encoded  
- Target: `TARGET` (binary: 0 = non-default, 1 = default)

## Data Cleaning & Feature Engineering

### Imputation Strategies

| Category | Action | Rationale |
|---------|--------|-----------|
| Low Missingness (≤1%) | Median Imputation | Preserves distribution |
| Moderate Missingness (10–20%) | Mean Imputation | Features are normalized |
| Bureau Features | Impute with 0 + Missing Flags | Behavioral signal |
| High Missingness (>50%) | Median + Flags | Absence may be predictive |
| Categorical (Light) | Mode Imputation | Preserves category distribution |
| Categorical (Heavy) | "Unknown" + Flags | Treats missingness as signal |
| String Normalization | `.str.lower().strip()` | Prevents encoding duplication |

Additional steps:
- Dropped `SK_ID_CURR` (non-predictive)
- Created binary flags for missingness
- Preserved all rows (no `dropna()`)

## Class Balancing

Original default rate: **8.07%**  
Balanced using undersampling:

```python
non_default = df[df['TARGET'] == 0]
default = df[df['TARGET'] == 1]
non_default_sample = resample(non_default, replace=False, n_samples=len(default), random_state=42)
balanced_df = pd.concat([non_default_sample, default])

- Final shape: (49,650, 181)
- Encoded with pd.get_dummies(drop_first=True)
- Final feature count: 293

Dimensionality Reduction with PCA
- PCA applied after scaling
- Tested components: 140, 180, 200
- Best performance observed with 180 components
- PCA improved generalization and reduced overfitting
pca = PCA(n_components=180)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca = PCA(n_components=180)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

Deep Learning Model
model = keras.Sequential([
    layers.Input(shape=(180,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


- Optimizer: Adam (lr = 0.0005)
- Loss: Binary Crossentropy
- Metrics: Accuracy, AUC
- EarlyStopping: monitor='val_auc', patience=3

Final Evaluation (PCA = 200)

| Metric | Value | 
| Accuracy | 0.68 | 
| Recall | 0.69 | 
| AUC | 0.74 | 

Confusion Matrix
[[TN FP]
 [FN TP]]

Classification Report
| Class | Precision | Recall | F1-score | 
| 0 | 0.68 | 0.67 | 0.68 | 
| 1 | 0.69 | 0.69 | 0.69 | 

Stratified K-Fold Validation
- Folds: 5
- PCA Components Tested: 140, 180, 200
- Metrics: Accuracy, Recall, AUC, Val_Loss
Cross-Validated Results
| PCA | Accuracy | Recall | AUC | Val_Loss | 
| 140 | 0.67 | 0.64 | 0.73 | 0.61 | 
| 180 | 0.68 | 0.69 | 0.74 | 0.60 | 
| 200 | 0.68 | 0.67 | 0.74 | 0.60 | 

Model & Pipeline Saving
dump(scaler, 'scaler.pkl')
dump(pca, 'pca_180.pkl')
model_final.save('model_final_180.keras')

Reflections
- PCA improved model stability and interpretability
- High recall makes this model suitable for risk-sensitive applications
- Stratified K-Fold validation confirmed generalization across folds
- Future work: explore SMOTE, ensemble models, and deployment pipelines

Comparison: V2 vs V3

| Feature | V2 | V3 |  What's New in V3 | 
| Data Cleaning | Strategic imputations and flags | Same | No change | 
| Row Preservation | No rows dropped | Same | Both avoid dropna() | 
| Encoding | pd.get_dummies(drop_first=True) | Same | Identical encoding | 
| Class Balancing | Undersampling via resample() | Same | Same technique | 
| Model Architecture | 64 → Dropout → 32 → Sigmoid | 128 → Dropout → 64 → 32 → Sigmoid | Deeper and more expressive in V3 | 
| Scaler | StandardScaler() | Same | Same normalization | 
| PCA |  Not used |  Applied with 180 components | Dimensionality reduction added | 
| Model Input | Scaled data | PCA-transformed data | More compact and informative input in V3 | 
| EarlyStopping | Monitored val_auc | Same (mode='max' added) | Slight tuning improvement | 
| Evaluation | Accuracy, Recall, AUC, Confusion Matrix | Same + ROC Curve | ROC visualization added | 
| Cross-Validation |  Not implemented |  Stratified K-Fold (5 folds) | Robustness testing added | 
| PCA Tuning |  |  Tested 140, 180, 200 components | Dimensionality impact explored | 
| Model Saving | Not specified | Saves scaler.pkl, pca_180.pkl, model_final_180.keras | Ready for reuse and deployment | 
| Final Report | Metrics and reflections | Detailed documentation + plots | More complete and visual reporting | 

Summary
Version V3 builds on the solid foundation of V2 and introduces:
- PCA for dimensionality reduction
- Stratified K-Fold cross-validation
- Deeper neural network architecture
- Full pipeline saving for reuse
- Enhanced reporting with visualizations

These upgrades make V3 more robust, production-ready, and easier to maintain or share.






