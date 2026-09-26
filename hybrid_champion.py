import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data.astype(np.float32)
y = data.target.astype(np.float32)

print("=" * 60)
print("HYBRID QUANTUM-CLASSICAL DISEASE PREDICTION")
print("=" * 60)

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 5. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 6. QUANTUM CIRCUIT SETTINGS
# ============================================================\=

num_qubits = 4

print(f"\nNumber of qubits: {num_qubits}")


# ============================================================
# 7. QUANTUM PARAMETERS
# ============================================================

# 4 input parameters:
# x0, x1, x2, x3

input_params = ParameterVector(
    "x",
    num_qubits
)


# 3 variational layers
#
# Each layer:
# 4 RY parameters
# 4 RZ parameters
#
# 8 parameters per layer
#
# 3 × 8 = 24 trainable parameters

weight_params = ParameterVector(
    "theta",
    num_qubits * 6
)


# ============================================================
# 8. CREATE QUANTUM CIRCUIT
# ============================================================

qc = QuantumCircuit(num_qubits)


# ============================================================
# 9. DATA ENCODING
# ============================================================

for i in range(num_qubits):

    qc.ry(
        input_params[i],
        i
    )

    qc.rz(
        input_params[i],
        i
    )


# ============================================================
# 10. VARIATIONAL LAYER 1
# ============================================================

for i in range(num_qubits):

    qc.ry(
        weight_params[i],
        i
    )

    qc.rz(
        weight_params[num_qubits + i],
        i
    )


# ============================================================
# 11. ENTANGLEMENT 1
# ============================================================

for i in range(num_qubits - 1):

    qc.cx(
        i,
        i + 1
    )


# ============================================================
# 12. VARIATIONAL LAYER 2
# ============================================================

offset = 2 * num_qubits

for i in range(num_qubits):

    qc.ry(
        weight_params[offset + i],
        i
    )

    qc.rz(
        weight_params[offset + num_qubits + i],
        i
    )


# ============================================================
# 13. ENTANGLEMENT 2
# ============================================================

for i in range(num_qubits - 1):

    qc.cx(
        i,
        i + 1
    )


# ============================================================
# 14. VARIATIONAL LAYER 3
# ============================================================

offset = 4 * num_qubits

for i in range(num_qubits):

    qc.ry(
        weight_params[offset + i],
        i
    )

    qc.rz(
        weight_params[offset + num_qubits + i],
        i
    )


# ============================================================
# 15. ENTANGLEMENT 3
# ============================================================

for i in range(num_qubits - 1):

    qc.cx(
        i,
        i + 1
    )


print("\nQuantum circuit created.")
print(f"Qubits: {num_qubits}")
print("Variational layers: 3")
print("Trainable quantum parameters: 24")


# ============================================================
# 16. QUANTUM OBSERVABLES
# ============================================================

observables = []

for i in range(num_qubits):

    pauli = ["I"] * num_qubits

    pauli[i] = "Z"

    observable = SparsePauliOp.from_list(
        [
            (
                "".join(pauli),
                1.0
            )
        ]
    )

    observables.append(observable)


# ============================================================
# 17. CREATE QUANTUM NEURAL NETWORK
# ============================================================

qnn = EstimatorQNN(
    circuit=qc,
    observables=observables,
    input_params=list(input_params),
    weight_params=list(weight_params)
)


# ============================================================
# 18. HYBRID MODEL
# ============================================================

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # CLASSICAL FEATURE EXTRACTION
        # ----------------------------------------------------

        self.classical1 = nn.Linear(
            30,
            16
        )

        self.classical2 = nn.Linear(
            16,
            4
        )


        # ----------------------------------------------------
        # QUANTUM NEURAL NETWORK
        # ----------------------------------------------------

        self.quantum = TorchConnector(
            qnn
        )


        # ----------------------------------------------------
        # HYBRID FUSION
        #
        # Classical features = 4
        # Quantum features   = 4
        #
        # Combined = 8
        # ----------------------------------------------------

        self.fusion = nn.Linear(
            8,
            4
        )


        # ----------------------------------------------------
        # FINAL CLASSIFIER
        # ----------------------------------------------------

        self.output = nn.Linear(
            4,
            1
        )


    def forward(self, x):

        # ====================================================
        # CLASSICAL PROCESSING
        # ====================================================

        x = torch.relu(
            self.classical1(x)
        )

        classical_features = torch.tanh(
            self.classical2(x)
        )


        # ====================================================
        # CONVERT CLASSICAL FEATURES
        # INTO QUANTUM ROTATION ANGLES
        # ====================================================

        quantum_input = (
            classical_features * np.pi
        )


        # ====================================================
        # QUANTUM PROCESSING
        # ====================================================

        quantum_features = self.quantum(
            quantum_input
        )


        # ====================================================
        # HYBRID FUSION
        # ====================================================

        combined = torch.cat(
            (
                classical_features,
                quantum_features
            ),
            dim=1
        )


        # ====================================================
        # CLASSICAL DECISION LAYER
        # ====================================================

        x = torch.relu(
            self.fusion(combined)
        )

        x = self.output(x)


        return x


# ============================================================
# 19. CREATE MODEL
# ============================================================

model = HybridModel()

print("\nHybrid model created successfully.")


# ============================================================
# 20. LOSS FUNCTION
# ============================================================

loss_function = nn.BCEWithLogitsLoss()


# ============================================================
# 21. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003,
    weight_decay=1e-4
)


# ============================================================
# 22. TRAINING SETTINGS
# ============================================================

epochs = 15
batch_size = 32

num_samples = len(X_train)


print("\n")
print("=" * 60)
print("TRAINING HYBRID MODEL")
print("=" * 60)

print(f"Epochs     : {epochs}")
print(f"Batch size : {batch_size}")


# ============================================================
# 23. TRAINING LOOP
# ============================================================

for epoch in range(epochs):

    model.train()

    permutation = torch.randperm(
        num_samples
    )

    total_loss = 0.0


    # --------------------------------------------------------
    # MINI-BATCH TRAINING
    # --------------------------------------------------------

    for i in range(
        0,
        num_samples,
        batch_size
    ):

        indices = permutation[
            i:i + batch_size
        ]

        batch_X = X_train[
            indices
        ]

        batch_y = y_train[
            indices
        ]


        # Clear gradients

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        predictions = model(
            batch_X
        )


        # ----------------------------------------------------
        # CALCULATE LOSS
        # ----------------------------------------------------

        loss = loss_function(
            predictions,
            batch_y
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE PARAMETERS
        # ----------------------------------------------------

        optimizer.step()


        total_loss += (
            loss.item()
            * len(batch_X)
        )


    # ========================================================
    # TRAINING ACCURACY
    # ========================================================

    model.eval()

    with torch.no_grad():

        train_logits = model(
            X_train
        )

        train_probabilities = torch.sigmoid(
            train_logits
        )

        train_predictions = (
            train_probabilities >= 0.5
        ).float()

        train_accuracy = (
            train_predictions == y_train
        ).float().mean()


    average_loss = (
        total_loss /
        num_samples
    )


    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"| Loss: {average_loss:.4f} "
        f"| Train Accuracy: "
        f"{train_accuracy.item() * 100:.2f}%"
    )


# ============================================================
# 24. FINAL TESTING
# ============================================================

print("\n")
print("=" * 60)
print("TESTING FINAL MODEL")
print("=" * 60)


model.eval()

with torch.no_grad():

    test_logits = model(
        X_test
    )

    test_probabilities = torch.sigmoid(
        test_logits
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).float()


# ============================================================
# 25. CONVERT RESULTS
# ============================================================

y_true = (
    y_test
    .numpy()
    .flatten()
)

y_pred = (
    test_predictions
    .numpy()
    .flatten()
)


# ============================================================
# 26. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# 27. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RESIDUAL HYBRID QUANTUM-CLASSICAL RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 28. COMPARE AGAINST RANDOM FOREST
# ============================================================

random_forest_accuracy = 92.11

print(
    f"\nRandom Forest baseline: "
    f"{random_forest_accuracy:.2f}%"
)

print(
    f"Hybrid QNN accuracy: "
    f"{accuracy * 100:.2f}%"
)


difference = (
    accuracy * 100
    - random_forest_accuracy
)


if difference > 0:

    print(
        f"\n🔥 HYBRID MODEL BEAT "
        f"THE RANDOM FOREST BY "
        f"{difference:.2f} percentage points!"
    )

else:

    print(
        f"\nHybrid model is "
        f"{abs(difference):.2f} percentage points "
        f"below the Random Forest."
    )


print("\nExperiment complete.")