import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import NBeats

# TRAIN_END = "2012-12-31 23:00:00"
# VAL_END = "2013-12-31 23:00:00"


backcast_len = 24 * 7
forecast_len = 24

HIDDEN_SIZE = 256
THETA_SIZE = 128
NUMBER_OF_BLOCKS = 6
NUMBER_OF_LAYERS = 4

BATCH_SIZE = 256
EPOCHS = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
STRIDE = 6
TEST_RATIO = 0.20
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
        stride,
    ):
        self.series = series
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.windows = []

        total_length = backcast_len + forecast_len

        for i, values in enumerate(series):
            n = len(values)
            last_possible_start = n - total_length

            for start in range(0, last_possible_start + 1, stride):
                self.windows.append((i, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        series_index, start = self.windows[idx]
        values = self.series[series_index]
        backcast_end = start + self.backcast_len
        forecast_end = backcast_end + self.forecast_len

        x = values[start:backcast_end]
        x = torch.tensor(x, dtype=torch.float32)

        y = values[backcast_end:forecast_end]
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def load_uci_from_path(path):

    dataframe = pd.read_csv(path, sep=";")
    timestamp_column = dataframe.columns[0]
    dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column])
    dataframe = dataframe.set_index(timestamp_column)
    dataframe = dataframe.apply(pd.to_numeric,errors="coerce")
        
    hourly_data = dataframe.resample("1h").mean()
    missing_data = hourly_data.isna().mean()
    
    good_columns = missing_data < 0.01
    hourly_data = hourly_data.loc[:, good_columns]
    
    hourly_data = hourly_data.interpolate(
        method="time",
        limit_direction="both",
    )
    
    hourly_data = hourly_data.ffill()
    hourly_data = hourly_data.bfill()

    return hourly_data

def prepare_series(hourly_data, test_ratio):

    number_of_rows = len(hourly_data)
    train_size = int(number_of_rows * (1.0 - test_ratio))

    train_df = hourly_data.iloc[:train_size]
    test_df = hourly_data.iloc[train_size:]

    train_series = []
    test_series = []

    statistics = {}

    for column in hourly_data.columns:

        train_values = train_df[column].to_numpy(dtype=np.float32)
        test_values = test_df[column].to_numpy(dtype=np.float32)

        mean = float(np.mean(train_values))
        std = float(np.std(train_values))

        if std < 1e-6:
            continue

        train_normalized = (train_values - mean) / std
        test_normalized = (test_values - mean) / std

        train_series.append(train_normalized.astype(np.float32))
        test_series.append(test_normalized.astype(np.float32))

        statistics[column] = {
            "mean": mean,
            "std": std,
        }

    return (
        train_series,
        test_series,
        list(statistics.keys()),
        statistics,
    )

def train(args):

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    hourly = load_uci_from_path(args.data)
    ( train_series, test_series, series_names, statistics) = prepare_series(hourly,TEST_RATIO)

    dataset = WindowDataset( series=train_series, backcast_len=backcast_len, forecast_len=forecast_len, stride=STRIDE)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = NBeats(
        backcast_length=backcast_len,
        forecast_length=forecast_len,
        hidden_size=HIDDEN_SIZE,
        theta_size=THETA_SIZE,
        number_of_blocks=NUMBER_OF_BLOCKS,
        number_of_layers=NUMBER_OF_LAYERS,
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
        patience=4,
    )
    huber_loss = torch.nn.HuberLoss(delta=1.0)
    best_loss = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()
        total_loss = 0.0
        count = 0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = huber_loss(prediction,y)
            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        epoch_loss = total_loss / count
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch:03d} loss={epoch_loss:.2f} learning_rate={optimizer.param_groups[0]['lr']:.2e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "backcast_len": backcast_len,
                    "forecast_len": forecast_len,
                    "hidden_size": HIDDEN_SIZE,
                    "theta_size": THETA_SIZE,
                    "n_blocks": NUMBER_OF_BLOCKS,
                    "n_layers": NUMBER_OF_LAYERS,
                },
                "series_names": series_names,
                "statistics": statistics,
                "best_train_loss": best_loss,
                "dataset": "UCI Electricity Load Diagrams 2011-2014",
                "frequency": "hourly",
                "forecast_horizon": "24 hours",
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(args.output)),
                exist_ok=True,
            )

            torch.save(
                checkpoint,
                args.output,
            )

    return (
        model,
        test_series,
        series_names,
        statistics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="./electricityloaddiagrams20112014/LD2011_2014.txt"
    )
    parser.add_argument("--output", default="./uci_nbeats.pt")
    args = parser.parse_args()
    train(args)