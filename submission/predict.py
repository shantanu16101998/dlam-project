import argparse
import os

import numpy as np
import pandas as pd
import torch

from model import NBeats


def load_checkpoint(path, device):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["model_config"]

    model = NBeats(
        backcast_length=config["backcast_length"],
        forecast_length=config["forecast_length"],
        hidden_size=config["hidden_size"],
        theta_size=config["theta_size"],
        n_blocks=config["n_blocks"],
        n_layers=config["n_layers"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


def make_prediction(
    model,
    history,
    mean,
    std,
    device,
):
    """
    Predict the complete 336-hour horizon.
    """

    backcast_length = model.backcast_length
    forecast_length = model.forecast_length

    history = np.asarray(
        history,
        dtype=np.float32,
    )

    if len(history) < backcast_length:
        raise ValueError(
            f"Not enough history: "
            f"{len(history)} < {backcast_length}"
        )

    # Use the final 28 days.
    x = history[-backcast_length:]

    # Normalize exactly as during training.
    x = (
        x - mean
    ) / std

    x = torch.tensor(
        x,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(x)

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Back to original target scale.
    prediction = (
        prediction * std
        + mean
    )

    return prediction


def main(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_path = os.path.join(
        args.input_dir,
        "train.csv",
    )

    forecast_path = os.path.join(
        args.input_dir,
        "forecast_index_validation.csv",
    )

    print("Loading:", train_path)
    print("Loading:", forecast_path)

    train = pd.read_csv(
        train_path
    )

    forecast_index = pd.read_csv(
        forecast_path
    )

    train["timestamp"] = pd.to_datetime(
        train["timestamp"]
    )

    forecast_index["timestamp"] = pd.to_datetime(
        forecast_index["timestamp"]
    )

    train = train.sort_values(
        ["series_id", "timestamp"]
    ).reset_index(drop=True)

    forecast_index = forecast_index.sort_values(
        ["series_id", "timestamp"]
    ).reset_index(drop=True)

    model, checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    statistics = checkpoint["statistics"]

    predictions = []

    for series_id, forecast_group in forecast_index.groupby(
        "series_id",
        sort=False,
    ):

        forecast_group = forecast_group.sort_values(
            "timestamp"
        )

        history_df = train[
            train["series_id"] == series_id
        ].sort_values(
            "timestamp"
        )

        history = (
            history_df["target"]
            .astype(float)
            .to_numpy()
        )

        if series_id not in statistics:
            raise ValueError(
                f"Series {series_id} "
                f"not present in checkpoint."
            )

        mean = statistics[series_id]["mean"]
        std = statistics[series_id]["std"]

        horizon = len(forecast_group)

        # N-BEATS was trained for 336-step forecasts.
        if horizon != model.forecast_length:
            raise ValueError(
                f"{series_id}: expected "
                f"{model.forecast_length} forecast rows, "
                f"got {horizon}"
            )

        prediction = make_prediction(
            model=model,
            history=history,
            mean=mean,
            std=std,
            device=device,
        )

        for timestamp, value in zip(
            forecast_group["timestamp"],
            prediction,
        ):

            predictions.append(
                {
                    "series_id": series_id,
                    "timestamp": timestamp,
                    "target": float(value),
                }
            )

    result = pd.DataFrame(
        predictions
    )

    # Restore the exact ordering of forecast_index.
    result = forecast_index[
        ["series_id", "timestamp"]
    ].merge(
        result,
        on=["series_id", "timestamp"],
        how="left",
    )

    if result["target"].isna().any():
        raise RuntimeError(
            "Some predictions are missing."
        )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(args.output_file)
        ),
        exist_ok=True,
    )

    result.to_csv(
        args.output_file,
        index=False,
    )

    print(
        f"Wrote {len(result):,} predictions "
        f"to {args.output_file}"
    )

    print(
        result.head()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()

    main(args)