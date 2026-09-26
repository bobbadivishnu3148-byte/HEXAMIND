import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
FOLDS = 5

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.003

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

data = load_breast_cancer()

X = data.data.astype(np.float32)
y = data.target.astype(np.float32)

print("=" * 60)
print("HYBRID QNN 5-FOLD VALIDATION")
print("=" * 60)

print(f"Dataset samples : {len(X)}")
print(f"Features        : {X.shape[1]}")
print(f"Folds           : {FOLDS}")
print()


# ============================================================
# CREATE QUANTUM CIRCUIT
# ============================================================

num_qubits = 4

input_params = ParameterVector("x", num_qubits)

# 3 variational layers
# Each layer = 4 RY + 4 RZ = 8 parameters
# 3 × 8 = 24 parameters
weight_params = ParameterVector(
    "theta",
    num_qubits * 6
)

qc = QuantumCircuit(num_qubits)


# ------------------------------------------------------------
# Quantum input encoding
# ------------------------------------------------------------

for i in range(num_qubits):
    qc.ry(input_params[i], i)
    qc.rz(input_params[i], i)


# ------------------------------------------------------------
# Variational layer 1
# ------------------------------------------------------------

for i in range(num_qubits):
    qc.ry(weight_params[i], i)
    qc.rz(weight_params[num_qubits + i], i)

for i in range(num_qubits - 1):
    qc.cx(i, i + 1)


# ------------------------------------------------------------
# Variational layer 2
# ------------------------------------------------------------

offset = 2 * num_qubits

for i in range(num_qubits):
    qc.ry(weight_params[offset + i], i)
    qc.rz(weight_params[offset + num_qubits + i], i)

for i in range(num_qubits - 1):
    qc.cx(i, i + 1)


# ------------------------------------------------------------
# Variational layer 3
# ------------------------------------------------------------

offset = 4 * num_qubits

for i in range(num_qubits):
    qc.ry(weight_params[offset + i], i)
    qc.rz(weight_params[offset + num_qubits + i], i)

for i in range(num_qubits - 1):
    qc.cx(i, i + 1)


# ============================================================
# QUANTUM OBSERVABLES
# ============================================================

observables = []

for i in range(num_qubits):

    pauli = ["I"] * num_qubits
    pauli[i] = "Z"

    observable = SparsePauliOp.from_list(
        [("".join(pauli), 1.0)]
    )

    observables.append(observable)


# ============================================================
# CREATE QNN
# ============================================================

qnn = EstimatorQNN(
    circuit=qc,
    observables=observables,
    input_params=list(input_params),
    weight_params=list(weight_params),
    input_gradients=True
)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Classical encoder
        self.classical1 = nn.Linear(30, 16)
        self.classical2 = nn.Linear(16, 4)

        # Quantum layer
        self.quantum = TorchConnector(qnn)

        # Fusion layer
        self.fusion = nn.Linear(8, 4)

        # Final classifier
        self.output = nn.Linear(4, 1)


    def forward(self, x):

        # Classical feature extraction
        x = torch.relu(
            self.classical1(x)
        )

        classical_features = torch.tanh(
            self.classical2(x)
        )

        # Convert classical features
        # into quantum rotation angles
        quantum_input = classical_features * np.pi

        # Quantum processing
        quantum_features = self.quantum(
            quantum_input
        )

        # Classical + quantum fusion
        combined = torch.cat(
            (
                classical_features,
                quantum_features
            ),
            dim=1
        )

        x = torch.relu(
            self.fusion(combined)
        )

        x = self.output(x)

        return x


# ============================================================
# METRIC STORAGE
# ============================================================

hybrid_results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

rf_results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=SEED
)


for fold, (train_idx, test_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print()
    print("=" * 60)
    print(f"FOLD {fold}/{FOLDS}")
    print("=" * 60)


    # ========================================================
    # SPLIT
    # ========================================================

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]


    # ========================================================
    # SCALE
    # ========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )


    rf_accuracy = accuracy_score(
        y_test,
        rf_pred
    )

    rf_precision = precision_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_pred,
        zero_division=0
    )


    rf_results["accuracy"].append(
        rf_accuracy
    )

    rf_results["precision"].append(
        rf_precision
    )

    rf_results["recall"].append(
        rf_recall
    )

    rf_results["f1"].append(
        rf_f1
    )


    # ========================================================
    # CONVERT TO PYTORCH
    # ========================================================

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    ).reshape(-1, 1)

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32
    ).reshape(-1, 1)


    # ========================================================
    # CREATE FRESH HYBRID MODEL
    # ========================================================

    # Different seed for each fold
    torch.manual_seed(
        SEED + fold
    )

    model = HybridModel()


    # ========================================================
    # LOSS
    # ========================================================

    loss_function = nn.BCEWithLogitsLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )


    # ========================================================
    # TRAIN
    # ========================================================

    num_samples = len(
        X_train_tensor
    )


    for epoch in range(EPOCHS):

        model.train()

        permutation = torch.randperm(
            num_samples
        )

        total_loss = 0.0


        for i in range(
            0,
            num_samples,
            BATCH_SIZE
        ):

            indices = permutation[
                i:i + BATCH_SIZE
            ]

            batch_X = X_train_tensor[
                indices
            ]

            batch_y = y_train_tensor[
                indices
            ]


            optimizer.zero_grad()


            predictions = model(
                batch_X
            )


            loss = loss_function(
                predictions,
                batch_y
            )


            loss.backward()


            optimizer.step()


            total_loss += (
                loss.item()
                * len(batch_X)
            )


        average_loss = (
            total_loss / num_samples
        )


        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )


    # ========================================================
    # HYBRID TEST
    # ========================================================

    model.eval()

    with torch.no_grad():

        test_logits = model(
            X_test_tensor
        )

        test_probabilities = torch.sigmoid(
            test_logits
        )

        test_predictions = (
            test_probabilities >= 0.5
        ).float()


    y_true = y_test_tensor.numpy().flatten()

    y_pred = test_predictions.numpy().flatten()


    hybrid_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    hybrid_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    hybrid_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    hybrid_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    hybrid_results["accuracy"].append(
        hybrid_accuracy
    )

    hybrid_results["precision"].append(
        hybrid_precision
    )

    hybrid_results["recall"].append(
        hybrid_recall
    )

    hybrid_results["f1"].append(
        hybrid_f1
    )


    # ========================================================
    # FOLD RESULTS
    # ========================================================

    print()
    print("Fold results:")
    print()

    print(
        f"Random Forest Accuracy : "
        f"{rf_accuracy * 100:.2f}%"
    )

    print(
        f"Hybrid QNN Accuracy    : "
        f"{hybrid_accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Random Forest F1       : "
        f"{rf_f1 * 100:.2f}%"
    )

    print(
        f"Hybrid QNN F1          : "
        f"{hybrid_f1 * 100:.2f}%"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print()
print("=" * 60)
print("FINAL 5-FOLD VALIDATION RESULTS")
print("=" * 60)


def print_results(name, results):

    print()
    print(name)
    print("-" * 40)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:

        values = np.array(
            results[metric]
        )

        mean = values.mean() * 100
        std = values.std() * 100

        print(
            f"{metric.capitalize():10s}: "
            f"{mean:.2f}% ± {std:.2f}%"
        )


print_results(
    "RANDOM FOREST",
    rf_results
)

print_results(
    "HYBRID QNN",
    hybrid_results
)


# ============================================================
# ACCURACY COMPARISON
# ============================================================

rf_mean = (
    np.mean(
        rf_results["accuracy"]
    ) * 100
)

hybrid_mean = (
    np.mean(
        hybrid_results["accuracy"]
    ) * 100
)

difference = hybrid_mean - rf_mean


print()
print("=" * 60)
print("ACCURACY COMPARISON")
print("=" * 60)

print(
    f"Random Forest mean accuracy : "
    f"{rf_mean:.2f}%"
)

print(
    f"Hybrid QNN mean accuracy    : "
    f"{hybrid_mean:.2f}%"
)

print(
    f"Difference                  : "
    f"{difference:+.2f} percentage points"
)

print()


if difference > 0:

    print(
        "🏆 HYBRID QNN OUTPERFORMED "
        "RANDOM FOREST"
    )

else:

    print(
        "Random Forest performed "
        "better in cross-validation."
    )


print()
print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)