import argparse
import numpy as np
import pandas as pd
import torch
from model import NBeats

BACKCAST_LENGTH = 168
FORECAST_LENGTH = 24


def mae(true_value, predicted_value):
    return np.mean(np.abs(true_value - predicted_value))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    denominator = np.maximum(
        np.abs(y_true),
        1e-6,
    )
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denominator))


def load_data(path):
    df = pd.read_csv(
        path,
        sep=";",
    )
    timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    hourly = df.resample("1h").mean()
    missing_fraction = hourly.isna().mean()
    valid_columns = missing_fraction[missing_fraction < 0.01].index
    hourly = hourly[valid_columns]
    hourly = hourly.interpolate(
        method="time",
        limit_direction="both",
    )
    hourly = hourly.ffill().bfill()
    return hourly


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Device:",
        device,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["model_config"]
    model = NBeats(
        backcast_length=config["backcast_len"],
        forecast_length=config["forecast_len"],
        hidden_size=config["hidden_size"],
        theta_size=config["theta_size"],
        number_of_blocks=config["n_blocks"],
        number_of_layers=config["n_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hourly = load_data(args.data)
    n = len(hourly)
    split = int(n * 0.80)

    train_df = hourly.iloc[:split]
    test_df = hourly.iloc[split:]

    series_names = checkpoint["series_names"]
    statistics = checkpoint["statistics"]

    all_true = []
    all_pred = []

    print("Evaluating:", len(series_names), "series")

    for col in series_names:
        train_values = train_df[col].to_numpy(dtype=np.float32)
        test_values = test_df[col].to_numpy(dtype=np.float32)

        mean = statistics[col]["mean"]
        std = statistics[col]["std"]

        train_normalized = (train_values - mean) / std
        test_normalized = (test_values - mean) / std

        history = list(train_normalized)

        predictions = []
        actuals = []

        for start in range(
            0,
            len(test_normalized),
            FORECAST_LENGTH,
        ):
            y_true = test_normalized[start : start + FORECAST_LENGTH]

            if len(y_true) < FORECAST_LENGTH:
                break

            x = np.asarray(
                history[-BACKCAST_LENGTH:],
                dtype=np.float32,
            )
            x_tensor = (
                torch.tensor(
                    x,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                pred = model(x_tensor)

            pred = pred.squeeze(0).cpu().numpy()

            predictions.extend(pred)
            actuals.extend(y_true)
            history.extend(y_true)

        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        predictions_original = predictions * std + mean
        actuals_original = actuals * std + mean

        all_pred.extend(predictions_original)
        all_true.extend(actuals_original)

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    result_mae = mae(
        all_true,
        all_pred,
    )
    result_rmse = rmse(
        all_true,
        all_pred,
    )
    result_mape = mape(
        all_true,
        all_pred,
    )

    print("UCI N-BEATS RESULTS")
    print(f"MAE  : {result_mae:.6f}")
    print(f"RMSE : {result_rmse:.6f}")
    print(f"MAPE : {result_mape:.4f}%")

    results = pd.DataFrame(
        {
            "metric": [
                "MAE",
                "RMSE",
                "MAPE",
            ],
            "value": [
                result_mae,
                result_rmse,
                result_mape,
            ],
        }
    )
    results.to_csv(
        args.output,
        index=False,
    )
    print(
        "Saved results:",
        args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="./electricityloaddiagrams20112014/LD2011_2014.txt",
    )
    parser.add_argument(
        "--checkpoint",
        default="./uci_nbeats.pt",
    )
    parser.add_argument(
        "--output",
        default="./uci_results.csv",
    )
    args = parser.parse_args()
    evaluate(args)
