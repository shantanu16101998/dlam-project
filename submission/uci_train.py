import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import NBeats


# ============================================================
# Configuration
# ============================================================

# UCI data is converted from 15-minute to hourly.
#
# 7 days of hourly history
BACKCAST_LENGTH = 24 * 7       # 168 hours

# Predict the next 24 hours
FORECAST_LENGTH = 24

HIDDEN_SIZE = 256
THETA_SIZE = 128
N_BLOCKS = 6
N_LAYERS = 4

BATCH_SIZE = 256
EPOCHS = 2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Every 6 hours create a new training window.
STRIDE = 6

# Last 20% of the data is held out chronologically.
TEST_RATIO = 0.20

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):

    def __init__(
        self,
        series,
        backcast_length,
        forecast_length,
        stride,
    ):

        self.series = series
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.windows = []

        total_length = (
            backcast_length +
            forecast_length
        )

        for series_idx, values in enumerate(series):

            n = len(values)

            max_start = n - total_length

            if max_start < 0:
                continue

            for start in range(
                0,
                max_start + 1,
                stride,
            ):

                self.windows.append(
                    (series_idx, start)
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        series_idx, start = self.windows[idx]

        values = self.series[series_idx]

        x = values[
            start:
            start + self.backcast_length
        ]

        y = values[
            start + self.backcast_length:
            start +
            self.backcast_length +
            self.forecast_length
        ]

        return (
            torch.tensor(
                x,
                dtype=torch.float32,
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
            ),
        )


# ============================================================
# Load and preprocess UCI
# ============================================================

def load_uci(path):

    print("=" * 70)
    print("Loading UCI Electricity Load Diagrams")
    print("=" * 70)

    # UCI file is semicolon separated.
    df = pd.read_csv(
        path,
        sep=";",
    )

    print(
        "Raw shape:",
        df.shape,
    )

    # First column is timestamp.
    timestamp_col = df.columns[0]

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col]
    )

    df = df.set_index(
        timestamp_col
    )

    # Convert client columns to numeric.
    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # 15-minute -> hourly
    # --------------------------------------------------------

    print("Converting 15-minute data to hourly...")

    hourly = df.resample("1h").mean()

    print(
        "Hourly shape:",
        hourly.shape,
    )

    # --------------------------------------------------------
    # Remove columns with excessive missing data.
    # --------------------------------------------------------

    missing_fraction = hourly.isna().mean()

    valid_columns = (
        missing_fraction[
            missing_fraction < 0.01
        ].index
    )

    hourly = hourly[
        valid_columns
    ]

    print(
        "Usable clients:",
        len(hourly.columns),
    )

    # --------------------------------------------------------
    # Fill small gaps.
    # --------------------------------------------------------

    hourly = hourly.interpolate(
        method="time",
        limit_direction="both",
    )

    # Any remaining missing values:
    hourly = hourly.ffill().bfill()

    return hourly


# ============================================================
# Create train/test series
# ============================================================

def prepare_series(
    hourly,
    test_ratio,
):

    n = len(hourly)

    split = int(
        n * (1.0 - test_ratio)
    )

    train_df = hourly.iloc[
        :split
    ]

    test_df = hourly.iloc[
        split:
    ]

    print()
    print(
        "Total hourly observations:",
        n,
    )

    print(
        "Training observations:",
        len(train_df),
    )

    print(
        "Test observations:",
        len(test_df),
    )

    train_series = []
    test_series = []

    statistics = {}

    for col in hourly.columns:

        train_values = (
            train_df[col]
            .to_numpy(dtype=np.float32)
        )

        test_values = (
            test_df[col]
            .to_numpy(dtype=np.float32)
        )

        mean = float(
            np.mean(train_values)
        )

        std = float(
            np.std(train_values)
        )

        if std < 1e-6:
            continue

        # IMPORTANT:
        # Normalize using TRAINING statistics only.
        train_normalized = (
            train_values - mean
        ) / std

        test_normalized = (
            test_values - mean
        ) / std

        train_series.append(
            train_normalized.astype(
                np.float32
            )
        )

        test_series.append(
            test_normalized.astype(
                np.float32
            )
        )

        statistics[col] = {
            "mean": mean,
            "std": std,
        }

    return (
        train_series,
        test_series,
        list(statistics.keys()),
        statistics,
    )


# ============================================================
# Train
# ============================================================

def train(args):

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Load UCI
    # --------------------------------------------------------

    hourly = load_uci(
        args.data
    )

    (
        train_series,
        test_series,
        series_names,
        statistics,
    ) = prepare_series(
        hourly,
        TEST_RATIO,
    )

    # --------------------------------------------------------
    # Training windows
    # --------------------------------------------------------

    dataset = WindowDataset(
        series=train_series,
        backcast_length=BACKCAST_LENGTH,
        forecast_length=FORECAST_LENGTH,
        stride=STRIDE,
    )

    print()
    print(
        "Training windows:",
        len(dataset),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NBeats(
        backcast_length=BACKCAST_LENGTH,
        forecast_length=FORECAST_LENGTH,
        hidden_size=HIDDEN_SIZE,
        theta_size=THETA_SIZE,
        n_blocks=N_BLOCKS,
        n_layers=N_LAYERS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    loss_fn = torch.nn.HuberLoss(
        delta=1.0
    )

    best_loss = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        count = 0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(x)

            loss = loss_fn(
                prediction,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = x.size(0)

            running_loss += (
                loss.item() *
                batch_size
            )

            count += batch_size

        epoch_loss = (
            running_loss / count
        )

        scheduler.step(
            epoch_loss
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={epoch_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "model_config": {
                    "backcast_length":
                        BACKCAST_LENGTH,

                    "forecast_length":
                        FORECAST_LENGTH,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "theta_size":
                        THETA_SIZE,

                    "n_blocks":
                        N_BLOCKS,

                    "n_layers":
                        N_LAYERS,
                },

                "series_names":
                    series_names,

                "statistics":
                    statistics,

                "best_train_loss":
                    best_loss,

                "dataset":
                    "UCI Electricity Load Diagrams 2011-2014",

                "frequency":
                    "hourly",

                "forecast_horizon":
                    "24 hours",
            }

            os.makedirs(
                os.path.dirname(
                    os.path.abspath(
                        args.output
                    )
                ),
                exist_ok=True,
            )

            torch.save(
                checkpoint,
                args.output,
            )

            print(
                "Saved:",
                args.output,
            )

    print()
    print(
        "Training complete."
    )

    return (
        model,
        test_series,
        series_names,
        statistics,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="../electricityloaddiagrams20112014/LD2011_2014.txt",
    )

    parser.add_argument(
        "--output",
        default="./uci_nbeats.pt",
    )

    args = parser.parse_args()

    train(args)
    