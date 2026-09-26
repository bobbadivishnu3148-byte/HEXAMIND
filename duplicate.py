import streamlit as st
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Hybrid QML Health Platform",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Hybrid Quantum Machine Learning Platform")
st.caption("Early Disease Detection — Research Prototype")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def get_data():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y, data


X, y, data = get_data()


# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("🏥 Medical Dataset")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Samples", X.shape[0])

with c2:
    st.metric("Original Features", X.shape[1])

with c3:
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

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# PCA
# 30 FEATURES → 4 FEATURES
# =========================================================

pca = PCA(n_components=4)

X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)


st.header("🔬 Data Preprocessing")

st.info(
    f"Standardization completed. "
    f"PCA reduced the feature space from "
    f"{X.shape[1]} features to 4 quantum-compatible features."
)


# =========================================================
# RANDOM FOREST
# =========================================================

st.header("🌳 Classical Machine Learning")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_reduced, y_train)

rf_predictions = rf.predict(X_test_reduced)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)


# =========================================================
# QUANTUM MACHINE LEARNING
# =========================================================

st.header("⚛️ Quantum Machine Learning")

num_qubits = 4

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=1
)

ansatz = RealAmplitudes(
    num_qubits=num_qubits,
    reps=1
)

optimizer = COBYLA(
    maxiter=20
)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)


# Train only when button is pressed

if st.button("🚀 Train Hybrid QML Model"):

    with st.spinner(
        "Training the Variational Quantum Classifier..."
    ):

        vqc.fit(
            X_train_reduced,
            y_train
        )

        q_predictions = vqc.predict(
            X_test_reduced
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

    st.success("Quantum model training completed!")


    st.header("📊 Model Performance")


    col1, col2 = st.columns(2)


    # RANDOM FOREST

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


    # QML

    with col2:

        st.subheader("⚛️ Quantum VQC")

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
    # QUANTUM CIRCUIT
    # =====================================================

    st.header("🔌 Quantum Circuit")

    quantum_circuit = feature_map.compose(ansatz)

    st.code(
        quantum_circuit.draw(output="text"),
        language="text"
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.header("📈 Quantum Model Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        q_predictions
    )

    st.dataframe(
        cm,
        use_container_width=True
    )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.header("🔍 Feature Importance")

    importance = rf.feature_importances_

    feature_names = [
        "Quantum Feature 1",
        "Quantum Feature 2",
        "Quantum Feature 3",
        "Quantum Feature 4"
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


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application is a research prototype for "
    "machine-learning experimentation and is not a medical "
    "diagnostic system."
)