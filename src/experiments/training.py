
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.data_processing.config import RANDOM_STATE, CLASS_NAMES
from src.models.pytorch_model import BearingMLP


def set_random_seed(seed=RANDOM_STATE):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(
            output,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X_batch.size(0)
        )

        predictions = output.argmax(dim=1)

        correct += (
            (predictions == y_batch)
            .sum()
            .item()
        )

        total += X_batch.size(0)

    if total == 0:
        raise ValueError("Training loader contains no samples.")

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    verbose=True
):
    """
    Train a PyTorch model for multiple epochs.

    Returns
    -------
    dict
        Training loss and accuracy history.
    """

    history = {
        "train_loss": [],
        "train_accuracy": []
    }

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_accuracy:.4f}"
            )

    return history


def train_sklearn_model(
    model,
    X_train,
    y_train
):
    """
    Train a Scikit-learn model.

    Scaling is handled internally by the model Pipeline,
    when the selected model requires it.
    """

    model.fit(
        X_train,
        y_train
    )

    return model


def train_pytorch_model(
    X_train,
    y_train,
    fold,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    hidden_features=50,
    verbose=True
):
    """
    Prepare and train the PyTorch MLP for one CV fold.

    The scaler is fitted exclusively with training data.

    Returns
    -------
    dict
        Trained model, scaler, training history and device.
    """

    # Reproducibility
    set_random_seed(RANDOM_STATE + fold)

    # Convert input data to NumPy arrays
    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int64
    )

    # Validate input dimensions
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")

    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different lengths.")

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values.")

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")

    # Fit scaler using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    # Convert data to PyTorch tensors
    X_tensor = torch.from_numpy(X_train_scaled)
    y_tensor = torch.from_numpy(y_train)

    # Create training dataset
    train_dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    # Reproducible DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE + fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    # Select computational device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build neural network
    model = BearingMLP(
        input_features=X_train.shape[1],
        hidden_features=hidden_features,
        num_classes=len(CLASS_NAMES)
    )

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Train neural network
    history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        verbose=verbose
    )

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "device": device
    }


def predict_pytorch_model(
    model,
    scaler,
    X_test,
    device
):
    """
    Generate predictions using a trained PyTorch model.

    The test data is transformed using the scaler
    fitted on the training data.
    """

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    if X_test.ndim != 2:
        raise ValueError("X_test must be a 2D array.")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or infinite values.")

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    X_tensor = torch.from_numpy(
        X_test_scaled
    ).to(device)

    model.eval()

    with torch.no_grad():

        output = model(X_tensor)

        predictions = output.argmax(
            dim=1
        )

    return predictions.cpu().numpy()
