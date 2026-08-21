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

HIDDEN_LAYER_SIZE = 256
THETA_SIZE = 128
NUMBER_OF_BLOCKS = 6
NUMBER_OF_LAYERS = 4
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42

WAPE_WEIGHT = 0.8
HUBER_WEIGHT = 0.2

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

        for i, values in enumerate(series):
            n = len(values)
            last_possible_start = n - total_length
            for start in range(0, last_possible_start + 1, stride):
                self.windows.append((i, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index_position):
        series_index, start = self.windows[index_position]
        values = self.series[series_index]
        backcast_end = start + self.backcast_len
        forecast_end = backcast_end + self.forecast_len
        
        x = values[start : backcast_end]
        x = torch.tensor(x, dtype=torch.float32)
        
        y = values[backcast_end : forecast_end]
        y = torch.tensor(y, dtype=torch.float32)
        
        return x,y, series_index


def load_training_data(path):
    dataframe = pd.read_csv(path, parse_dates=["timestamp"])
    dataframe = dataframe.sort_values(["series_id", "timestamp"])
    dataframe = dataframe.reset_index(drop=True)

    series_ids = sorted(dataframe["series_id"].unique())
    series = []
    statistics = {}

    for series_id in series_ids:
        series_id_df = dataframe[dataframe["series_id"] == series_id]
        series_id_df = series_id_df.sort_values("timestamp")
        
        values = series_id_df["target"].astype(float).to_numpy()

        mean = float(values.mean())
        std_dev = float(values.std())
    
        if std_dev < 1e-6:
            std_dev = 1.0

        normalized = (values - mean) / std_dev
        series.append(normalized.astype(np.float32))

        statistics[series_id] = {"mean": mean, "std": std_dev,}

    return series_ids, series, statistics

def original_scale_wape(prediction, target, series_indices, means, standard_deviations,):
    batch_means = means[series_indices].unsqueeze(1)
    batch_stds = standard_deviations[series_indices].unsqueeze(1)

    prediction_original = prediction * batch_stds + batch_means
    target_original = target * batch_stds + batch_means

    absolute_error = torch.abs(prediction_original - target_original).sum()
    absolute_target = torch.abs(target_original).sum()

    wape = absolute_error / absolute_target.clamp_min(1e-6)

    return wape, absolute_error, absolute_target


def train(args):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series_ids, series, statistics = load_training_data(args.train)

    print("Number of series: ", len(series_ids))
    print("Observations per series: ", len(series[0]))

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
        num_workers=0
    )

    model = NBeats(
        backcast_length=backcast_len,
        forecast_length=forecast_len,
        hidden_size=HIDDEN_LAYER_SIZE,
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
        patience=3,
    )

    huber_loss = torch.nn.HuberLoss(delta=1.0)

    means = torch.tensor(
        [statistics[series_id]["mean"] for series_id in series_ids],
        dtype=torch.float32,
        device=device,
    )
    standard_deviations = torch.tensor(
        [statistics[series_id]["std"] for series_id in series_ids],
        dtype=torch.float32,
        device=device,
    )

    best_wape = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        count = 0
        total_absolute_error = 0.0
        total_absolute_target = 0.0
        

        for x, y, series_indices in loader:
            x = x.to(device)
            y = y.to(device)
            series_indices = series_indices.to(device)

            optimizer.zero_grad()
            prediction = model(x)

            wape_loss, error_sum, target_sum = original_scale_wape(
                prediction,
                y,
                series_indices,
                means,
                standard_deviations,
            )
            
            stable_loss = huber_loss( prediction, y)
            loss = (WAPE_WEIGHT * wape_loss + HUBER_WEIGHT * stable_loss)
            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            
            optimizer.step()
            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            count += batch_size
            total_absolute_error += error_sum.detach().item()
            total_absolute_target += target_sum.detach().item()            

        epoch_loss = total_loss / count
        epoch_wape = (100.0 * total_absolute_error / max(total_absolute_target, 1e-6))
        scheduler.step(epoch_loss)

        print(
            f"Epoch {epoch} loss={epoch_loss:.6f} WAPE={epoch_wape:.4f}% learning_rate={optimizer.param_groups[0]['lr']:.5f}"
        )

        if epoch_wape < best_wape:
            best_wape = epoch_wape

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "backcast_length": backcast_len,
                    "forecast_length": forecast_len,
                    "hidden_size": HIDDEN_LAYER_SIZE,
                    "theta_size": THETA_SIZE,
                    "n_blocks": NUMBER_OF_BLOCKS,
                    "n_layers": NUMBER_OF_LAYERS,
                },
                "series_ids": series_ids,
                "statistics": statistics,
                "best_train_wape": best_wape,
                "best_train_loss": epoch_loss,
                "loss_config": {
                    "wape_weight": WAPE_WEIGHT,
                    "huber_weight": HUBER_WEIGHT,
                },
            }

            torch.save(checkpoint, args.output)
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
