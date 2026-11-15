
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: #1B2330;
        border: 1px solid #303B4B;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #AAB7C4;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #FFFFFF;
    }

    .subtitle {
        color: #AAB7C4;
        font-size: 16px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #E2E8F0;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. PROJECT PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

LOGISTIC_MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    BASE_DIR / "models" / "random_forest.joblib"
)

# IMPORTANT: Define THRESHOLD BEFORE COMPARISON_MODELS.
THRESHOLD = 0.70

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

TARGET = "Class"

# Model thresholds selected during validation experiments.
COMPARISON_MODELS = {
    "Logistic Regression": {
        "path": LOGISTIC_MODEL_PATH,
        "threshold": 0.50
    },
    "Random Forest": {
        "path": RANDOM_FOREST_MODEL_PATH,
        "threshold": 0.55
    },
    "XGBoost": {
        "path": MODEL_PATH,
        "threshold": THRESHOLD
    }
}


# ==========================================
# 4. MODEL AND DATA LOADING
# ==========================================

@st.cache_resource
def load_model(path_string, modified_time):
    return joblib.load(path_string)


@st.cache_resource
def load_scaler(path_string, modified_time):
    return joblib.load(path_string)


@st.cache_data
def load_test_data(path_string, modified_time):
    return pd.read_csv(path_string)


def get_model(path):

    if not path.exists():
        return None

    try:
        return load_model(
            str(path),
            path.stat().st_mtime
        )

    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None


def get_scaler():

    if not SCALER_PATH.exists():
        st.error(
            f"Scaler file not found: {SCALER_PATH}"
        )
        return None

    try:
        return load_scaler(
            str(SCALER_PATH),
            SCALER_PATH.stat().st_mtime
        )

    except Exception as error:
        st.error(f"Scaler loading error: {error}")
        return None


def get_test_data():

    if not TEST_PATH.exists():
        st.error(
            f"Test dataset not found: {TEST_PATH}"
        )
        return None

    try:
        return load_test_data(
            str(TEST_PATH),
            TEST_PATH.stat().st_mtime
        )

    except Exception as error:
        st.error(f"Test dataset error: {error}")
        return None


# ==========================================
# 5. LOAD MAIN MODEL
# ==========================================

model = get_model(MODEL_PATH)
scaler = get_scaler()


# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "PR-AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "Confusion Matrix": confusion_matrix(
            y_true,
            predictions
        ),

        "Predictions": predictions
    }


def prepare_uploaded_data(data):

    missing = [
        col for col in FEATURES
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Class is deliberately excluded from inference.
    X = data[FEATURES].copy()

    for col in FEATURES:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    if X.isnull().any().any():
        raise ValueError(
            "Dataset contains missing or non-numeric values."
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Dataset contains infinite values."
        )

    if scaler is None:
        raise ValueError(
            "Scaler is not available. Check models/scaler.joblib."
        )

    # Only Time and Amount were scaled during training.
    X.loc[:, ["Time", "Amount"]] = scaler.transform(
        X[["Time", "Amount"]]
    )

    return X


def create_results(original_data, probabilities):

    results = original_data[FEATURES].copy()

    results.insert(
        0,
        "Transaction ID",
        np.arange(1, len(results) + 1)
    )

    results["Fraud Probability"] = probabilities

    results["Prediction"] = np.where(
        probabilities >= THRESHOLD,
        "Fraud",
        "Legitimate"
    )

    results["Risk Level"] = pd.cut(
        probabilities,
        bins=[
            -0.000001,
            0.20,
            0.50,
            0.80,
            1.000001
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    ).astype(str)

    return results


def create_confusion_matrix(matrix, title):

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],
        y=[
            "Actual Legitimate",
            "Actual Fraud"
        ],
        color_continuous_scale="Blues",
        title=title
    )

    fig.update_layout(
        template="plotly_dark"
    )

    return fig


# ==========================================
# 7. HEADER
# ==========================================

st.markdown(
    """
    <div class="main-title">
        🛡️ AI-Powered Credit Card Fraud Detection
    </div>

    <div class="subtitle">
        Intelligent Fraud Detection using Machine Learning,
        XGBoost and Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# ==========================================
# 8. SIDEBAR
# ==========================================

st.sidebar.title("🛡️ Fraud Detection")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Model Performance",
        "Model Comparison",
        "Upload & Predict",
        "Advanced Fraud Analytics",
        "Transaction Analytics",
        "Transaction Explorer",
        "SHAP Explainability"
    ]
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Decision Threshold",
    f"{THRESHOLD:.2f}"
)

st.sidebar.caption(
    "Powered by XGBoost and SHAP"
)


# ==========================================
# 9. SESSION STATE
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = None

if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = None


# ==========================================
# 10. OVERVIEW
# ==========================================

if page == "Overview":

    st.title("Project Overview")

    st.write(
        """
        This AI-powered application detects potentially
        fraudulent credit card transactions using machine
        learning classification models.

        The system combines XGBoost, Random Forest,
        Logistic Regression and SHAP explainability.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Machine Learning",
        "XGBoost"
    )

    col2.metric(
        "Input Features",
        "30"
    )

    col3.metric(
        "Explainable AI",
        "SHAP"
    )

    col4.metric(
        "Threshold",
        "70%"
    )

    st.markdown("### Main Features")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            - Fraud prediction
            - Transaction risk classification
            - Model performance evaluation
            - Model comparison
            """
        )

    with c2:

        st.markdown(
            """
            - Advanced fraud analytics
            - Interactive transaction explorer
            - SHAP explanations
            - CSV export
            """
        )

    st.info(
        "V1–V28 are anonymized PCA components from the "
        "benchmark dataset, not directly interpretable "
        "business variables."
    )


# ==========================================
# 11. MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("XGBoost Model Performance")

    test_df = get_test_data()

    if test_df is not None and model is not None:

        missing = [
            c for c in FEATURES + [TARGET]
            if c not in test_df.columns
        ]

        if missing:
            st.error(
                f"Test dataset is missing columns: {missing}"
            )

        else:

            X_test = test_df[FEATURES]
            y_test = test_df[TARGET].astype(int)

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            metrics = calculate_metrics(
                y_test,
                probabilities,
                THRESHOLD
            )

            st.markdown("### Evaluation Metrics")

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Precision",
                f"{metrics['Precision']:.4f}"
            )

            c2.metric(
                "Recall",
                f"{metrics['Recall']:.4f}"
            )

            c3.metric(
                "F1-Score",
                f"{metrics['F1-Score']:.4f}"
            )

            c4.metric(
                "ROC-AUC",
                f"{metrics['ROC-AUC']:.4f}"
            )

            c5.metric(
                "PR-AUC",
                f"{metrics['PR-AUC']:.4f}"
            )

            st.caption(
                f"Test transactions: {len(test_df):,} | "
                f"Threshold: {THRESHOLD:.2f}"
            )

            left, right = st.columns(2)

            with left:

                st.plotly_chart(
                    create_confusion_matrix(
                        metrics["Confusion Matrix"],
                        "XGBoost Confusion Matrix"
                    ),
                    use_container_width=True
                )

            with right:

                fpr, tpr, _ = roc_curve(
                    y_test,
                    probabilities
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name="XGBoost ROC"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Baseline",
                        line=dict(dash="dash")
                    )
                )

                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            precision, recall, _ = precision_recall_curve(
                y_test,
                probabilities
            )

            pr_fig = go.Figure()

            pr_fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name="Precision-Recall"
                )
            )

            pr_fig.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template="plotly_dark"
            )

            st.plotly_chart(
                pr_fig,
                use_container_width=True
            )

            tn, fp, fn, tp = (
                metrics["Confusion Matrix"].ravel()
            )

            st.markdown("### Confusion Matrix Details")

            a, b, c, d = st.columns(4)

            a.metric("True Negatives", f"{tn:,}")
            b.metric("False Positives", f"{fp:,}")
            c.metric("False Negatives", f"{fn:,}")
            d.metric("True Positives", f"{tp:,}")

            st.warning(
                "These results describe the saved benchmark test set. "
                "They do not guarantee performance on real-world banking data."
            )


# ==========================================
# 12. MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":

    st.title("Model Comparison Dashboard")

    test_df = get_test_data()

    if test_df is not None:

        if model is None:
            st.warning(
                "The XGBoost model file is unavailable."
            )

        missing = [
            c for c in FEATURES + [TARGET]
            if c not in test_df.columns
        ]

        if missing:
            st.error(
                f"Test dataset is missing columns: {missing}"
            )

        else:

            y_test = test_df[TARGET].astype(int)

            comparison_results = []
            model_outputs = {}

            for name, config in COMPARISON_MODELS.items():

                comparison_model = get_model(
                    config["path"]
                )

                if comparison_model is None:

                    st.warning(
                        f"{name} model is unavailable: "
                        f"{config['path']}"
                    )

                    continue

                try:

                    probabilities = (
                        comparison_model.predict_proba(
                            test_df[FEATURES]
                        )[:, 1]
                    )

                    metrics = calculate_metrics(
                        y_test,
                        probabilities,
                        config["threshold"]
                    )

                    comparison_results.append(
                        {
                            "Model": name,
                            "Threshold": config["threshold"],
                            "Precision": metrics["Precision"],
                            "Recall": metrics["Recall"],
                            "F1-Score": metrics["F1-Score"],
                            "ROC-AUC": metrics["ROC-AUC"],
                            "PR-AUC": metrics["PR-AUC"]
                        }
                    )

                    model_outputs[name] = metrics

                except Exception as error:

                    st.error(
                        f"Error evaluating {name}: {error}"
                    )

            if comparison_results:

                comparison_df = pd.DataFrame(
                    comparison_results
                )

                comparison_df = comparison_df.sort_values(
                    "F1-Score",
                    ascending=False
                )

                st.markdown("### Performance Comparison")

                st.dataframe(
                    comparison_df.style.format(
                        {
                            "Threshold": "{:.2f}",
                            "Precision": "{:.4f}",
                            "Recall": "{:.4f}",
                            "F1-Score": "{:.4f}",
                            "ROC-AUC": "{:.4f}",
                            "PR-AUC": "{:.4f}"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                chart_df = comparison_df.melt(
                    id_vars=["Model"],
                    value_vars=[
                        "Precision",
                        "Recall",
                        "F1-Score",
                        "ROC-AUC",
                        "PR-AUC"
                    ],
                    var_name="Metric",
                    value_name="Score"
                )

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Model Performance Comparison",
                    text_auto=".3f"
                )

                fig.update_layout(
                    template="plotly_dark",
                    yaxis_range=[0, 1.05]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                best_model = comparison_df.iloc[0]

                st.success(
                    f"Best model by F1-Score: "
                    f"{best_model['Model']} "
                    f"({best_model['F1-Score']:.4f})"
                )

                st.markdown("### Individual Confusion Matrix")

                selected_model = st.selectbox(
                    "Select a model",
                    comparison_df["Model"].tolist()
                )

                st.plotly_chart(
                    create_confusion_matrix(
                        model_outputs[selected_model][
                            "Confusion Matrix"
                        ],
                        f"{selected_model} Confusion Matrix"
                    ),
                    use_container_width=True
                )

                st.warning(
                    "The saved test split has been used during "
                    "iterative model experiments. These results "
                    "should not be described as a completely "
                    "untouched independent final evaluation."
                )

            else:

                st.error(
                    "No models could be evaluated. "
                    "Check the model files in the models folder."
                )


# ==========================================
# 13. UPLOAD AND PREDICT
# ==========================================

elif page == "Upload & Predict":

    st.title("Upload Transactions & Detect Fraud")

    if model is None:

        st.error(
            "XGBoost model is not available. "
            "Check models/xgboost.joblib."
        )

    elif scaler is None:

        st.error(
            "Scaler is not available. "
            "Check models/scaler.joblib."
        )

    else:

        st.write(
            "Upload an original transaction CSV containing "
            "Time, V1–V28 and Amount."
        )

        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                original_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    f"Dataset loaded: {len(original_df):,} rows"
                )

                with st.expander("Dataset Preview"):

                    st.dataframe(
                        original_df.head(10),
                        use_container_width=True
                    )

                if st.button(
                    "Run Fraud Detection",
                    type="primary"
                ):

                    with st.spinner(
                        "Analyzing transactions..."
                    ):

                        scaled_df = prepare_uploaded_data(
                            original_df
                        )

                        probabilities = model.predict_proba(
                            scaled_df
                        )[:, 1]

                        results = create_results(
                            original_df,
                            probabilities
                        )

                        st.session_state.results = results

                        # Store the exact scaled feature data used
                        # for prediction, for SHAP explanations.
                        st.session_state.scaled_data = (
                            scaled_df.reset_index(drop=True)
                        )

                        st.success(
                            "Fraud detection completed successfully!"
                        )

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )

        if st.session_state.results is not None:

            st.info(
                "Predictions below refer to the most recently "
                "processed uploaded dataset."
            )

            st.markdown("### Prediction Results")

            st.dataframe(
                st.session_state.results.head(20),
                use_container_width=True
            )

            csv = st.session_state.results.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "Download Complete Results CSV",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )


# ==========================================
# 14. ADVANCED FRAUD ANALYTICS
# ==========================================

elif page == "Advanced Fraud Analytics":

    st.title("Advanced Fraud Analytics")

    results = st.session_state.results

    if results is None:

        st.info(
            "Upload a dataset and run predictions first."
        )

    else:

        fraud = results[
            results["Prediction"] == "Fraud"
        ]

        legitimate = results[
            results["Prediction"] == "Legitimate"
        ]

        total = len(results)
        fraud_count = len(fraud)
        legitimate_count = len(legitimate)

        fraud_rate = (
            fraud_count / total * 100
            if total > 0 else 0
        )

        average_probability = (
            results["Fraud Probability"].mean() * 100
            if total > 0 else 0
        )

        total_fraud_amount = fraud["Amount"].sum()

        average_fraud_amount = (
            fraud["Amount"].mean()
            if fraud_count > 0 else 0
        )

        st.markdown("### Key Fraud Indicators")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Transactions",
            f"{total:,}"
        )

        c2.metric(
            "Predicted Fraud",
            f"{fraud_count:,}"
        )

        c3.metric(
            "Predicted Legitimate",
            f"{legitimate_count:,}"
        )

        c4.metric(
            "Predicted Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        st.markdown("### Financial Risk Analysis")

        a, b, c = st.columns(3)

        a.metric(
            "Total Predicted Fraud Amount",
            f"{total_fraud_amount:,.2f}"
        )

        b.metric(
            "Average Predicted Fraud Amount",
            f"{average_fraud_amount:,.2f}"
        )

        c.metric(
            "Average Fraud Probability",
            f"{average_probability:.2f}%"
        )

        st.warning(
            "Predicted fraud transactions are model classifications, "
            "not independently confirmed fraud cases."
        )

        st.markdown("### Risk Level Distribution")

        risk_counts = (
            results["Risk Level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Transactions"
        ]

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Transactions",
            color="Risk Level",
            title="Transaction Risk Distribution",
            text_auto=True
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# 15. TRANSACTION ANALYTICS
# ==========================================

elif page == "Transaction Analytics":

    st.title("Transaction Analytics")

    results = st.session_state.results

    if results is None:

        st.info(
            "Upload a dataset and run predictions first."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            distribution = (
                results["Prediction"]
                .value_counts()
                .reset_index()
            )

            distribution.columns = [
                "Prediction",
                "Count"
            ]

            fig = px.pie(
                distribution,
                names="Prediction",
                values="Count",
                hole=0.55,
                title="Fraud vs Legitimate Transactions"
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.histogram(
                results,
                x="Fraud Probability",
                nbins=50,
                title="Fraud Probability Distribution"
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("### Transaction Amount Analysis")

        fraud = results[
            results["Prediction"] == "Fraud"
        ]

        legitimate = results[
            results["Prediction"] == "Legitimate"
        ]

        amount_data = pd.concat(
            [
                fraud.assign(Group="Predicted Fraud"),
                legitimate.assign(Group="Predicted Legitimate")
            ]
        )

        fig = px.box(
            amount_data,
            x="Group",
            y="Amount",
            color="Group",
            title="Transaction Amount by Prediction",
            points=False
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("### Top 10 Highest-Risk Transactions")

        top10 = results.nlargest(
            10,
            "Fraud Probability"
        )

        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 16. TRANSACTION EXPLORER
# ==========================================

elif page == "Transaction Explorer":

    st.title("Transaction Explorer")

    results = st.session_state.results

    if results is None:

        st.info(
            "Upload a dataset and run predictions first."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            prediction_filter = st.selectbox(
                "Prediction Type",
                [
                    "All",
                    "Fraud",
                    "Legitimate"
                ]
            )

        with col2:

            min_probability = st.slider(
                "Minimum Fraud Probability",
                0.0,
                1.0,
                0.0,
                0.01
            )

        with col3:

            page_size = st.selectbox(
                "Rows Per Page",
                [
                    25,
                    50,
                    100,
                    250,
                    500,
                    1000
                ],
                index=5
            )

        filtered = results.copy()

        if prediction_filter != "All":

            filtered = filtered[
                filtered["Prediction"] == prediction_filter
            ]

        filtered = filtered[
            filtered["Fraud Probability"] >= min_probability
        ]

        st.markdown(
            f"### Matching Transactions: {len(filtered):,}"
        )

        if len(filtered) == 0:

            st.warning(
                "No transactions match your filters."
            )

        else:

            max_page = max(
                1,
                int(np.ceil(len(filtered) / page_size))
            )

            selected_page = st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=1,
                step=1
            )

            start = (
                int(selected_page) - 1
            ) * page_size

            end = start + page_size

            page_df = filtered.iloc[start:end]

            st.dataframe(
                page_df,
                use_container_width=True,
                hide_index=True
            )

            csv = filtered.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "Download Filtered Transactions",
                data=csv,
                file_name="filtered_transactions.csv",
                mime="text/csv"
            )

            st.markdown("### Transaction Details")

            selected_id = st.selectbox(
                "Select Transaction ID",
                page_df["Transaction ID"].tolist()
            )

            selected = results[
                results["Transaction ID"] == selected_id
            ].iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Fraud Probability",
                f"{selected['Fraud Probability']:.2%}"
            )

            c2.metric(
                "Prediction",
                selected["Prediction"]
            )

            c3.metric(
                "Risk Level",
                selected["Risk Level"]
            )


# ==========================================
# 17. SHAP EXPLAINABILITY
# ==========================================

elif page == "SHAP Explainability":

    st.title("Explainable AI — SHAP")

    results = st.session_state.results
    scaled_data = st.session_state.scaled_data

    if model is None:

        st.error(
            "XGBoost model is not available."
        )

    elif results is None or scaled_data is None:

        st.info(
            "Upload a dataset and run predictions first."
        )

    else:

        st.write(
            """
            SHAP explains how individual input features contribute
            to the model's output. These explanations describe
            model behavior, not causal relationships.
            """
        )

        selected_id = st.selectbox(
            "Select Transaction to Explain",
            results["Transaction ID"].tolist(),
            key="shap_selected_transaction"
        )

        if st.button(
            "Generate SHAP Explanation",
            type="primary"
        ):

            try:

                row_index = int(selected_id) - 1

                selected_row = scaled_data.iloc[
                    [row_index]
                ]

                with st.spinner(
                    "Calculating SHAP values..."
                ):

                    explainer = shap.TreeExplainer(
                        model
                    )

                    shap_values = explainer(
                        selected_row
                    )

                    # Handle SHAP output with a class dimension.
                    if len(shap_values.values.shape) == 3:

                        values = shap_values.values[
                            :, :, 1
                        ]

                        base_values = np.asarray(
                            shap_values.base_values
                        )

                        if base_values.ndim == 2:
                            base_values = base_values[:, 1]

                        explanation = shap.Explanation(
                            values=values[0],
                            base_values=base_values[0],
                            data=selected_row.iloc[0].values,
                            feature_names=FEATURES
                        )

                    else:

                        explanation = shap_values[0]

                fig, ax = plt.subplots(
                    figsize=(12, 7)
                )

                shap.plots.waterfall(
                    explanation,
                    max_display=15,
                    show=False
                )

                st.pyplot(
                    plt.gcf(),
                    clear_figure=True
                )

                plt.close("all")

                st.success(
                    "SHAP explanation generated successfully."
                )

                st.caption(
                    "SHAP values are in the model output scale "
                    "used by the explainer. They are not necessarily "
                    "probability percentages."
                )

            except Exception as error:

                st.error(
                    f"SHAP explanation error: {error}"
                )


# ==========================================
# 18. FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#AAB7C4; padding:15px;">
        <b>AI-Powered Credit Card Fraud Detection System</b>
        <br>
        Machine Learning | XGBoost | SHAP | Streamlit
        <br>
        <small>
        Educational and portfolio project.
        Not intended for real-world financial decisions.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
