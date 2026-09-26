import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# ============================================================
# QUANTUM SETTINGS
# ============================================================

NUM_QUBITS = 4


# ============================================================
# QUANTUM PARAMETERS
# ============================================================

input_params = ParameterVector(
    "x",
    NUM_QUBITS
)

weight_params = ParameterVector(
    "theta",
    NUM_QUBITS * 6
)


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

qc = QuantumCircuit(NUM_QUBITS)


# ============================================================
# DATA ENCODING
# ============================================================

for i in range(NUM_QUBITS):

    qc.ry(
        input_params[i],
        i
    )

    qc.rz(
        input_params[i],
        i
    )


# ============================================================
# 3 VARIATIONAL QUANTUM LAYERS
# ============================================================

for layer in range(3):

    offset = layer * 2 * NUM_QUBITS

    # RY + RZ rotations
    for i in range(NUM_QUBITS):

        qc.ry(
            weight_params[offset + i],
            i
        )

        qc.rz(
            weight_params[offset + NUM_QUBITS + i],
            i
        )

    # Linear entanglement
    for i in range(NUM_QUBITS - 1):

        qc.cx(
            i,
            i + 1
        )


# ============================================================
# QUANTUM OBSERVABLES
# ============================================================

observables = []

for i in range(NUM_QUBITS):

    pauli = ["I"] * NUM_QUBITS

    pauli[i] = "Z"

    observables.append(
        SparsePauliOp.from_list(
            [
                (
                    "".join(pauli),
                    1.0
                )
            ]
        )
    )


# ============================================================
# QUANTUM NEURAL NETWORK
# ============================================================

qnn = EstimatorQNN(
    circuit=qc,
    observables=observables,
    input_params=list(input_params),
    weight_params=list(weight_params)
)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Classical feature extraction
        self.classical1 = nn.Linear(
            30,
            16
        )

        self.classical2 = nn.Linear(
            16,
            4
        )

        # Quantum neural network
        self.quantum = TorchConnector(
            qnn
        )

        # Classical + quantum fusion
        self.fusion = nn.Linear(
            8,
            4
        )

        # Final classifier
        self.output = nn.Linear(
            4,
            1
        )


    def forward(self, x):

        # Classical processing
        x = torch.relu(
            self.classical1(x)
        )

        classical_features = torch.tanh(
            self.classical2(x)
        )


        # Convert classical features
        # into quantum rotation angles

        quantum_input = (
            classical_features * np.pi
        )


        # Quantum processing

        quantum_features = self.quantum(
            quantum_input
        )


        # Combine classical + quantum features

        combined = torch.cat(
            (
                classical_features,
                quantum_features
            ),
            dim=1
        )


        # Fusion layer

        x = torch.relu(
            self.fusion(combined)
        )


        # Final prediction

        x = self.output(x)

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(weights_path):

    model = HybridModel()

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_probability(
    model,
    scaled_features
):

    x = torch.tensor(
        np.asarray(
            scaled_features,
            dtype=np.float32
        ).reshape(1, 30)
    )

    with torch.no_grad():

        logits = model(x)

        probability = torch.sigmoid(
            logits
        ).item()

    return probability