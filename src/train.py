from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent



# ============================================================
# Configuration
# ============================================================

CSV_PATH = PROJECT_ROOT / "data" / "input" / "train.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "submission" / "forecast_lstm.pt"

SEQUENCE_LENGTH = 48
HORIZON = 24

HIDDEN_SIZE = 64
NUM_LAYERS = 2

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

VAL_FRACTION = 0.2

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Model
# ============================================================

class ForecastModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 23,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 24,
    ) -> None:
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.fc = torch.nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)

        # Last LSTM layer
        hidden = hidden[-1]

        return self.fc(hidden)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["series_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# Fill missing feature values
# ============================================================

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in {
        "series_id",
        "timestamp",
        "target",
    }
]

# Fill missing values independently for each time series.
df[FEATURE_COLUMNS] = (
    df.groupby("series_id")[FEATURE_COLUMNS]
      .transform(lambda x: x.ffill().bfill())
)


# Verify that no NaNs remain in model inputs.
remaining_nans = df[FEATURE_COLUMNS].isna().sum()

if remaining_nans.any():
    print("Remaining NaNs:")
    print(remaining_nans[remaining_nans > 0])

    raise ValueError(
        "NaNs remain in feature columns after imputation."
    )

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["series_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# Define features
# ============================================================

EXCLUDE_COLUMNS = {
    "series_id",
    "timestamp",
    "target",
}

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDE_COLUMNS
]

TARGET_COLUMN = "target"

print("Features:")
for column in FEATURE_COLUMNS:
    print(f"  {column}")

print()
print(f"Number of features: {len(FEATURE_COLUMNS)}")
print(f"Sequence length:   {SEQUENCE_LENGTH}")
print(f"Forecast horizon:   {HORIZON}")


# ============================================================
# Chronological train/validation split
# ============================================================

train_parts = []
val_parts = []

for series_id, group in df.groupby("series_id"):
    group = group.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(group) * (1.0 - VAL_FRACTION))

    # Make sure both sets have enough observations
    if split_idx <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Series {series_id} is too short for the requested "
            f"sequence length."
        )

    train_parts.append(group.iloc[:split_idx])

    # Keep enough history for validation windows.
    #
    # This allows validation windows to use observations immediately
    # preceding the validation period as historical context.
    val_start = max(0, split_idx - SEQUENCE_LENGTH)

    val_parts.append(group.iloc[val_start:])


train_df = pd.concat(train_parts).reset_index(drop=True)
val_df = pd.concat(val_parts).reset_index(drop=True)


# ============================================================
# Normalize using TRAINING data only
# ============================================================

feature_mean = train_df[FEATURE_COLUMNS].mean()
feature_std = train_df[FEATURE_COLUMNS].std()

# Avoid division by zero for constant features.
feature_std = feature_std.replace(0, 1.0)

target_mean = train_df[TARGET_COLUMN].mean()
target_std = train_df[TARGET_COLUMN].std()

if target_std == 0:
    target_std = 1.0


def normalize_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    mean = feature_mean.to_numpy(dtype=np.float32)
    std = feature_std.to_numpy(dtype=np.float32)

    return (values - mean) / std


def normalize_target(frame: pd.DataFrame) -> np.ndarray:
    values = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

    return (values - target_mean) / target_std


# ============================================================
# Create sliding windows
# ============================================================

def make_windows(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    xs = []
    ys = []

    for series_id, group in frame.groupby("series_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        X = normalize_features(group)
        y = normalize_target(group)

        total_length = len(group)

        max_start = (
            total_length
            - SEQUENCE_LENGTH
            - HORIZON
            + 1
        )

        for start in range(max_start):
            x_start = start
            x_end = start + SEQUENCE_LENGTH

            y_start = x_end
            y_end = y_start + HORIZON

            xs.append(
                X[x_start:x_end]
            )

            ys.append(
                y[y_start:y_end]
            )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


X_train, y_train = make_windows(train_df)
X_val, y_val = make_windows(val_df)

print()
print("Training shape:")
print("  X:", X_train.shape)
print("  y:", y_train.shape)

print()
print("Validation shape:")
print("  X:", X_val.shape)
print("  y:", y_val.shape)


# ============================================================
# PyTorch datasets
# ============================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:", device)


# ============================================================
# Model
# ============================================================

model = ForecastModel(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=HORIZON,
).to(device)


# ============================================================
# Optimizer / loss
# ============================================================

criterion = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        prediction = model(X_batch)

        loss = criterion(
            prediction,
            y_batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = X_batch.size(0)

        train_loss_sum += (
            loss.item() * batch_size
        )

        train_count += batch_size

    train_loss = train_loss_sum / train_count

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            prediction = model(X_batch)

            loss = criterion(
                prediction,
                y_batch,
            )

            batch_size = X_batch.size(0)

            val_loss_sum += (
                loss.item() * batch_size
            )

            val_count += batch_size

    val_loss = val_loss_sum / val_count

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.6f} | "
        f"val_loss={val_loss:.6f}"
    )

    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "epoch": epoch,
            "best_val_loss": best_val_loss,

            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "output_size": HORIZON,

            "sequence_length": SEQUENCE_LENGTH,
            "horizon": HORIZON,

            "feature_columns": FEATURE_COLUMNS,

            "feature_mean": feature_mean.to_dict(),
            "feature_std": feature_std.to_dict(),

            "target_mean": float(target_mean),
            "target_std": float(target_std),
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

        print(
            f"  -> saved checkpoint: "
            f"{CHECKPOINT_PATH}"
        )


print()
print("Training complete.")
print("Best validation loss:", best_val_loss)
print("Checkpoint:", CHECKPOINT_PATH)