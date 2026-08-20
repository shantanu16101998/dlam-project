import argparse

import numpy as np
import pandas as pd
import torch

from model import NBeats


BACKCAST_LENGTH = 168
FORECAST_LENGTH = 24


# ============================================================
# Metrics
# ============================================================

def mae(y_true, y_pred):

    return np.mean(
        np.abs(
            y_true - y_pred
        )
    )


def rmse(y_true, y_pred):

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def mape(y_true, y_pred):

    # Avoid division by zero.
    denominator = np.maximum(
        np.abs(y_true),
        1e-6,
    )

    return 100.0 * np.mean(
        np.abs(
            (y_true - y_pred)
            / denominator
        )
    )


# ============================================================
# Load UCI
# ============================================================

def load_data(path):

    df = pd.read_csv(
        path,
        sep=";",
    )

    timestamp_col = df.columns[0]

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col]
    )

    df = df.set_index(
        timestamp_col
    )

    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Convert 15-minute → hourly.
    hourly = df.resample("1h").mean()

    # Remove problematic clients.
    missing_fraction = hourly.isna().mean()

    valid_columns = (
        missing_fraction[
            missing_fraction < 0.01
        ].index
    )

    hourly = hourly[
        valid_columns
    ]

    hourly = hourly.interpolate(
        method="time",
        limit_direction="both",
    )

    hourly = hourly.ffill().bfill()

    return hourly


# ============================================================
# Evaluate
# ============================================================

def evaluate(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint[
        "model_config"
    ]

    model = NBeats(
        backcast_length=
            config["backcast_length"],

        forecast_length=
            config["forecast_length"],

        hidden_size=
            config["hidden_size"],

        theta_size=
            config["theta_size"],

        n_blocks=
            config["n_blocks"],

        n_layers=
            config["n_layers"],
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    hourly = load_data(
        args.data
    )

    n = len(hourly)

    split = int(
        n * 0.80
    )

    train_df = hourly.iloc[
        :split
    ]

    test_df = hourly.iloc[
        split:
    ]

    series_names = checkpoint[
        "series_names"
    ]

    statistics = checkpoint[
        "statistics"
    ]

    all_true = []
    all_pred = []

    print()
    print(
        "Evaluating:",
        len(series_names),
        "series"
    )

    # --------------------------------------------------------
    # Forecast each series
    # --------------------------------------------------------

    for col in series_names:

        train_values = (
            train_df[col]
            .to_numpy(
                dtype=np.float32
            )
        )

        test_values = (
            test_df[col]
            .to_numpy(
                dtype=np.float32
            )
        )

        mean = statistics[col][
            "mean"
        ]

        std = statistics[col][
            "std"
        ]

        # Normalize using training statistics.
        train_normalized = (
            train_values - mean
        ) / std

        test_normalized = (
            test_values - mean
        ) / std

        # We forecast the test period in
        # non-overlapping 24-hour chunks.
        history = list(
            train_normalized
        )

        predictions = []
        actuals = []

        for start in range(
            0,
            len(test_normalized),
            FORECAST_LENGTH,
        ):

            y_true = test_normalized[
                start:
                start + FORECAST_LENGTH
            ]

            if len(y_true) < FORECAST_LENGTH:
                break

            # Last 168 hours.
            x = np.asarray(
                history[
                    -BACKCAST_LENGTH:
                ],
                dtype=np.float32,
            )

            x_tensor = torch.tensor(
                x,
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                pred = model(
                    x_tensor
                )

            pred = (
                pred.squeeze(0)
                .cpu()
                .numpy()
            )

            # Save normalized values.
            predictions.extend(
                pred
            )

            actuals.extend(
                y_true
            )

            # IMPORTANT:
            # For a true multi-day forecasting
            # evaluation, we don't feed the
            # predicted values back as history.
            #
            # Instead, use the actual observations
            # after each forecast period.
            history.extend(
                y_true
            )

        predictions = np.asarray(
            predictions
        )

        actuals = np.asarray(
            actuals
        )

        # Convert back to original kW scale.
        predictions_original = (
            predictions * std + mean
        )

        actuals_original = (
            actuals * std + mean
        )

        all_pred.extend(
            predictions_original
        )

        all_true.extend(
            actuals_original
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    all_true = np.asarray(
        all_true
    )

    all_pred = np.asarray(
        all_pred
    )

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

    print()
    print("=" * 70)
    print("UCI N-BEATS RESULTS")
    print("=" * 70)

    print(
        f"MAE  : {result_mae:.6f}"
    )

    print(
        f"RMSE : {result_rmse:.6f}"
    )

    print(
        f"MAPE : {result_mape:.4f}%"
    )

    print("=" * 70)

    # Save results.
    results = pd.DataFrame({
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
    })

    results.to_csv(
        args.output,
        index=False,
    )

    print(
        "Saved results:",
        args.output,
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
        "--checkpoint",
        default="./uci_nbeats.pt",
    )

    parser.add_argument(
        "--output",
        default="./uci_results.csv",
    )

    args = parser.parse_args()

    evaluate(args)