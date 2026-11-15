# Credit Card Fraud Detection System
### Machine Learning • XGBoost • Explainable AI • Streamlit

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using advanced classification algorithms, optimized decision thresholds, and Explainable AI techniques.

The project combines **XGBoost, Random Forest, Logistic Regression, SHAP, and Streamlit** to build an interactive fraud detection and analysis platform.

---

## Project Overview

Credit card fraud detection is a challenging financial classification problem due to the extremely imbalanced nature of transaction data.

In this project, multiple Machine Learning models are trained and evaluated to identify fraudulent transactions while minimizing false alarms.

The application provides an interactive environment where users can upload transaction datasets, detect suspicious activities, compare model performance, analyze transaction patterns, and understand predictions through SHAP explanations.

## Dashboard Screenshots

### 1. Model Performance

Performance evaluation dashboard displaying classification metrics and model evaluation results.

![Model Performance](assets/model_performance.png)

### 2. Model Comparison

Comparison of Logistic Regression, Random Forest, and XGBoost using multiple evaluation metrics and optimized decision thresholds.

![Model Comparison](assets/model_comparison.png)

---

## Key Features

- **Fraud Detection:** Identify potentially fraudulent credit card transactions.
- **Multiple ML Models:** Train and evaluate three classification algorithms.
- **Threshold Optimization:** Optimize prediction thresholds using validation data.
- **Model Comparison:** Compare models using multiple evaluation metrics.
- **Explainable AI:** Understand model decisions using SHAP.
- **Feature Importance:** Identify the features that contribute most to predictions.
- **Transaction Analytics:** Explore transaction distributions and fraud patterns.
- **Interactive Dashboard:** Visualize results through Streamlit and Plotly.
- **Dataset Upload:** Upload transaction CSV files for fraud prediction.
- **Risk Classification:** Categorize transactions according to their predicted fraud probability.

---

## Technology Stack

| Technology | Application |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine Learning and preprocessing |
| XGBoost | Gradient boosting |
| Random Forest | Ensemble classification |
| Logistic Regression | Baseline classification |
| SHAP | Explainable AI |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| Joblib | Model serialization |

---

## Dataset

**Source:** Kaggle – Credit Card Fraud Detection Dataset.

The dataset contains anonymized credit card transactions made by European cardholders.

### Dataset Information

| Property | Value |
|---|---:|
| Original Transactions | 284,807 |
| Features | 31 |
| Normal Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Missing Values | 0 |
| Duplicate Rows | 1,081 |

### Class Distribution

Fraudulent transactions represent approximately **0.17%** of the original dataset.

This severe class imbalance makes fraud detection particularly challenging.

### Feature Description

- `Time`: Seconds elapsed between each transaction and the first transaction.
- `Amount`: Transaction amount.
- `V1` to `V28`: Anonymized features obtained through PCA transformation.
- `Class`: Target variable (0 = Normal, 1 = Fraudulent).

---

## Data Preprocessing

The following preprocessing pipeline was implemented:

1. Dataset inspection and exploratory data analysis.
2. Detection and removal of exact duplicate rows.
3. Validation of missing and invalid values.
4. Stratified training and testing split.
5. Feature scaling using StandardScaler.
6. Preparation of processed datasets for model training.

### Duplicate Removal

After removing exact duplicate rows while retaining one copy:

| Property | Value |
|---|---:|
| Original Rows | 284,807 |
| Duplicate Rows Removed | 1,081 |
| Final Rows | 283,726 |

### Feature Scaling

StandardScaler was applied only to:

- `Time`
- `Amount`

The PCA-transformed features (`V1`–`V28`) were kept unchanged.

The scaler was fitted exclusively on the training data to prevent preprocessing leakage into the test set.

### Dataset Splitting

A stratified 80/20 split was used.

| Dataset | Samples |
|---|---:|
| Training | 226,980 |
| Testing | 56,746 |

---

## Machine Learning Models

Three Machine Learning algorithms were implemented.

### 1. Logistic Regression

Used as a baseline classification model with balanced class weights to address the highly imbalanced dataset.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

### 3. XGBoost

A gradient boosting algorithm configured with class weighting to improve its ability to identify fraudulent transactions.

The models were evaluated using the same test split.

---

## Model Performance & Results

The models were evaluated using their selected decision thresholds.

### Model Comparison

| Evaluation Metric | XGBoost | Random Forest | Logistic Regression |
|---|---:|---:|---:|
| Threshold | 0.70 | 0.55 | 0.50 |
| Precision | **93.59%** | 92.00% | 5.62% |
| Recall | 76.84% | 72.63% | **87.37%** |
| F1-Score | **84.39%** | 81.18% | 10.57% |
| ROC-AUC | **97.82%** | 96.24% | 96.58% |
| PR-AUC | **82.40%** | 80.48% | 67.19% |

### Best Model: XGBoost

XGBoost achieved the strongest overall performance among the evaluated models.

**Key Results:**

| Metric | Result |
|---|---:|
| Precision | 93.59% |
| Recall | 76.84% |
| F1-Score | 84.39% |
| ROC-AUC | 97.82% |
| PR-AUC | 82.40% |
| Decision Threshold | 0.70 |

### XGBoost Confusion Matrix

At the selected threshold of 0.70:

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 56,646 | 5 |
| Actual Fraud | 22 | 73 |

The model correctly identified 73 fraudulent transactions out of 95 actual fraudulent transactions in the test dataset.

It also produced only 5 false positive predictions.

### Model Interpretation

- **XGBoost:** Best overall balance between precision and recall.
- **Random Forest:** Competitive performance with strong precision.
- **Logistic Regression:** Highest recall, but significantly more false positives.

XGBoost was selected as the best-performing model based on its F1-Score, ROC-AUC, and PR-AUC.

---

## Explainable AI (SHAP)

To improve model transparency, SHAP (SHapley Additive exPlanations) was integrated into the project.

SHAP helps explain how individual features influence the predictions made by the XGBoost model.

The analysis was performed on a sample of 2,000 test transactions.

### Top Features by Mean Absolute SHAP Value

| Feature | Mean Absolute SHAP |
|---|---:|
| V14 | 2.3178 |
| V4 | 1.8060 |
| V12 | 1.2375 |
| V10 | 0.9028 |
| V11 | 0.8356 |
| V3 | 0.7660 |
| V8 | 0.6817 |
| V1 | 0.5437 |
| V28 | 0.4805 |
| V26 | 0.4802 |

### SHAP Visualizations

The project generates:

- SHAP Summary Plot
- SHAP Feature Importance Plot
- Individual Prediction Explanations

These visualizations provide insights into the features influencing the model's decisions.

**Important:** `V1`–`V28` are anonymized PCA-transformed features. Their importance does not establish direct real-world causes of fraudulent activity.

---

## Project Architecture

```text
Fraud-Detection-System/
│
├── assets/
│   ├── model_performance.png
│   └── model_comparison.png
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
│
├── notebooks/
│
├── reports/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── transactions_over_time.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_importance.csv
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── duplicate_analysis.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── random_forest.py
│   ├── threshold_tuning.py
│   ├── xgboost_training.py
│   ├── xgboost_threshold_tuning.py
│   ├── xgboost_final_evaluation.py
│   ├── feature_importance.py
│   └── shap_analysis.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Fraud-Detection-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the original dataset from Kaggle and place it inside:

```text
data/raw/creditcard.csv
```

### 5. Prepare Trained Models

The application requires the trained model files and scaler inside the `models/` directory.

Generate these files by executing the project's preprocessing and model training scripts.

The processed training and testing datasets are also required for the model evaluation dashboard.

### 6. Launch the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## Evaluation Metrics

The project uses the following metrics:

- **Precision:** Proportion of predicted fraudulent transactions that are actually fraudulent.
- **Recall:** Proportion of actual fraudulent transactions successfully detected.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Measures the model's ability to distinguish between normal and fraudulent transactions across thresholds.
- **PR-AUC:** Measures the Precision-Recall trade-off and is particularly useful for imbalanced classification problems.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

---

## Limitations

- The dataset contains anonymized PCA-transformed features.
- Evaluation results depend on the dataset split and selected decision thresholds.
- The project has not been validated on live banking transaction streams.
- The selected threshold may require recalibration for different operational environments.
- The test results should be treated as experimental rather than a guarantee of real-world performance.

---

## Future Improvements

- Implement cross-validation and out-of-fold threshold optimization.
- Evaluate performance on additional fraud datasets.
- Develop real-time fraud detection capabilities.
- Build a REST API using FastAPI.
- Add model monitoring and data drift detection.
- Implement automated model retraining.
- Deploy the application to a cloud environment.

---

## Disclaimer

This project is developed for educational, research, and portfolio purposes.

It is not a production-ready financial fraud detection system. Real-world deployment requires additional validation, security measures, monitoring, and regulatory compliance.

---

## Author

**Machine Learning & Artificial Intelligence Portfolio Project**

Focused on Machine Learning, Data Science, Fraud Detection, and Explainable AI.

