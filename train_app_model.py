import os

import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from app_model import HybridModel


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data.astype(
    np.float32
)

y = data.target.astype(
    np.float32
)


print("=" * 60)
print("HYBRID QML APPLICATION MODEL")
print("=" * 60)

print(
    f"Dataset shape: {X.shape}"
)

print(
    f"Number of features: {X.shape[1]}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# PYTORCH TENSORS
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
# CREATE MODEL
# ============================================================

model = HybridModel()


# ============================================================
# LOSS
# ============================================================

loss_function = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.003,

    weight_decay=1e-4
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

epochs = 15

batch_size = 32

num_samples = len(X_train)


print()
print("=" * 60)
print("TRAINING APPLICATION MODEL")
print("=" * 60)

print(
    f"Epochs     : {epochs}"
)

print(
    f"Batch size : {batch_size}"
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(epochs):

    model.train()

    permutation = torch.randperm(
        num_samples
    )

    total_loss = 0.0


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


        # Forward pass

        predictions = model(
            batch_X
        )


        # Loss

        loss = loss_function(
            predictions,
            batch_y
        )


        # Backpropagation

        loss.backward()


        # Update parameters

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
# FINAL TEST
# ============================================================

print()
print("=" * 60)
print("TESTING APPLICATION MODEL")
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
# CONVERT RESULTS
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
# METRICS
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
# RESULTS
# ============================================================

print()

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


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


torch.save(
    model.state_dict(),
    "model/hybrid_model.pth"
)


# Save scaler

joblib.dump(
    scaler,
    "model/scaler.pkl"
)


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "model/hybrid_model.pth"
)

print(
    "model/scaler.pkl"
)

print()
print(
    "Training complete."
)

print(
    "The API will NOT retrain the model."
)