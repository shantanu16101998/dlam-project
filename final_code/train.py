import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import NBeats

backcast_len = 672
forecast_len = 336

HIDDEN_SIZE = 256
THETA_SIZE = 128
N_BLOCKS = 6
N_LAYERS = 4
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):

    def __init__(
        self,
        series,
        backcast_len,
        forecast_len,
        stride=6,
    ):
        self.series = series
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.stride = stride

        self.windows = []

        total_length = backcast_len + forecast_len

        for series_idx, values in enumerate(series):
            n = len(values)
            max_start = n - total_length
            for start in range(
                0,
                max_start + 1,
                stride,
            ):
                self.windows.append((series_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        series_idx, start = self.windows[idx]
        values = self.series[series_idx]
        x = values[start : start + self.backcast_len]
        y = values[
            start + self.backcast_len : start + self.backcast_len + self.forecast_len
        ]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def load_training_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    series_ids = sorted(df["series_id"].unique())

    series = []
    statistics = {}

    for sid in series_ids:
        sub = df[df["series_id"] == sid].sort_values("timestamp")
        values = sub["target"].astype(float).to_numpy()

        if np.isnan(values).any():
            raise ValueError(f"Target contains NaN values in {sid}")

        mean = float(values.mean())
        std = float(values.std())

        if std < 1e-6:
            std = 1.0

        normalized = (values - mean) / std
        series.append(normalized.astype(np.float32))

        statistics[sid] = {
            "mean": mean,
            "std": std,
        }
    return (
        series_ids,
        series,
        statistics,
    )


def train(args):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    series_ids, series, statistics = load_training_data(args.train)

    print("Number of series:", len(series_ids))
    print("Observations per series:", len(series[0]))

    dataset = WindowDataset(
        series=series,
        backcast_len=backcast_len,
        forecast_len=forecast_len,
        stride=6,
    )
    print("Training windows:", len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = NBeats(
        backcast_length=backcast_len,
        forecast_length=forecast_len,
        hidden_size=HIDDEN_SIZE,
        theta_size=THETA_SIZE,
        number_of_blocks=N_BLOCKS,
        number_of_layers=N_LAYERS,
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

    loss_fn = torch.nn.HuberLoss(delta=1.0)
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
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

            running_loss += loss.item() * batch_size
            count += batch_size

        epoch_loss = running_loss / count
        scheduler.step(epoch_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={epoch_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "backcast_length": backcast_len,
                    "forecast_length": forecast_len,
                    "hidden_size": HIDDEN_SIZE,
                    "theta_size": THETA_SIZE,
                    "n_blocks": N_BLOCKS,
                    "n_layers": N_LAYERS,
                },
                "series_ids": series_ids,
                "statistics": statistics,
                "best_train_loss": best_loss,
            }

            torch.save(
                checkpoint,
                args.output,
            )

            print("Saved checkpoint:", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="./data/input/train.csv",
    )
    parser.add_argument(
        "--output",
        default="./final_code/checkpoint.pt",
    )
    args = parser.parse_args()
    train(args)
