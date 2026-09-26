import streamlit as st
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid QML - V2",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Hybrid Quantum Machine Learning Platform")
st.subheader("Early Disease Detection — V2 Experiment")

st.info(
    "This is a research prototype comparing a classical "
    "Random Forest model with a quantum Variational "
    "Quantum Classifier (VQC)."
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    data = load_breast_cancer()

    return data.data, data.target, data


X, y, data = load_data()


# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("🏥 Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Samples", X.shape[0])

with col2:
    st.metric("Original Features", X.shape[1])

with col3:
    st.metric("Classes", len(np.unique(y)))


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# STANDARDIZATION
# =========================================================

standard_scaler = StandardScaler()

X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)


# =========================================================
# PCA
# 30 FEATURES → 4 FEATURES
# =========================================================

pca = PCA(n_components=4)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# =========================================================
# QUANTUM FEATURE SCALING
# Scale PCA features to [-pi, pi]
# =========================================================

quantum_scaler = MinMaxScaler(
    feature_range=(-np.pi, np.pi)
)

X_train_quantum = quantum_scaler.fit_transform(
    X_train_pca
)

X_test_quantum = quantum_scaler.transform(
    X_test_pca
)


st.header("🔬 Data Preprocessing")

st.write(
    f"Original features: **{X.shape[1]}**"
)

st.write(
    "PCA features: **4**"
)

st.write(
    "Quantum feature range: **[-π, π]**"
)


# =========================================================
# RANDOM FOREST
# =========================================================

st.header("🌳 Classical Machine Learning")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train_pca,
    y_train
)

rf_predictions = rf.predict(
    X_test_pca
)


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions
)

rf_recall = recall_score(
    y_test,
    rf_predictions
)

rf_f1 = f1_score(
    y_test,
    rf_predictions
)


# =========================================================
# QUANTUM MODEL
# =========================================================

st.header("⚛️ Quantum Machine Learning")

num_qubits = 4


# Quantum feature map
feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=2
)


# Trainable quantum circuit
ansatz = RealAmplitudes(
    num_qubits=num_qubits,
    reps=1
)


# Classical optimizer
optimizer = COBYLA(
    maxiter=10
)


# VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)


if st.button("🚀 Train Improved QML Model"):

    with st.spinner(
        "Training improved Quantum Machine Learning model..."
    ):

        vqc.fit(
            X_train_quantum,
            y_train
        )

        q_predictions = vqc.predict(
            X_test_quantum
        )

        q_predictions = np.asarray(
            q_predictions
        ).astype(int).ravel()


    # =====================================================
    # QML METRICS
    # =====================================================

    q_accuracy = accuracy_score(
        y_test,
        q_predictions
    )

    q_precision = precision_score(
        y_test,
        q_predictions
    )

    q_recall = recall_score(
        y_test,
        q_predictions
    )

    q_f1 = f1_score(
        y_test,
        q_predictions
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.success(
        "Improved QML model training completed!"
    )


    st.header("📊 Model Performance")


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    with col1:

        st.subheader("🌳 Random Forest")

        st.metric(
            "Accuracy",
            f"{rf_accuracy * 100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{rf_precision * 100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{rf_recall * 100:.2f}%"
        )

        st.metric(
            "F1 Score",
            f"{rf_f1 * 100:.2f}%"
        )


    # -----------------------------------------------------
    # IMPROVED QML
    # -----------------------------------------------------

    with col2:

        st.subheader("⚛️ Improved QML")

        st.metric(
            "Accuracy",
            f"{q_accuracy * 100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{q_precision * 100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{q_recall * 100:.2f}%"
        )

        st.metric(
            "F1 Score",
            f"{q_f1 * 100:.2f}%"
        )


    # =====================================================
    # IMPROVEMENT FROM V1
    # =====================================================

    st.header("📈 QML Improvement")

    st.write(
        "V1 QML accuracy: **51.75%**"
    )

    improvement = (
        q_accuracy * 100
    ) - 51.75

    st.metric(
        "Change from V1",
        f"{improvement:+.2f} percentage points"
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.header("🔲 QML Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        q_predictions
    )

    st.dataframe(
        cm,
        use_container_width=True
    )


    # =====================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # =====================================================

    st.header("🔍 Feature Importance")

    importance = rf.feature_importances_

    feature_names = [
        "PCA Feature 1",
        "PCA Feature 2",
        "PCA Feature 3",
        "PCA Feature 4"
    ]

    importance_data = {
        "Feature": feature_names,
        "Importance": importance
    }

    st.bar_chart(
        importance_data,
        x="Feature",
        y="Importance"
    )


    # =====================================================
    # QUANTUM CIRCUIT
    # =====================================================

    st.header("🔌 Quantum Circuit")

    quantum_circuit = feature_map.compose(
        ansatz
    )

    st.code(
        quantum_circuit.draw(
            output="text"
        ),
        language="text"
    )


# =========================================================
# PIPELINE
# =========================================================

st.divider()

st.header("🔄 Current Pipeline")

st.write(
    """
    Medical Dataset
    ↓
    Standardization
    ↓
    PCA: 30 → 4 Features
    ↓
    Quantum Feature Scaling
    ↓
    ┌─────────────────┬─────────────────┐
    │                 │                 │
    🌳 Random Forest   ⚛️ VQC
    │                 │
    └─────────────────┴─────────────────┘
                    ↓
             Performance
               Comparison
    """
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ Research prototype only. This application is "
    "not a medical diagnostic system and should not be "
    "used for clinical decisions."
)