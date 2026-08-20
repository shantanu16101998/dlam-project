import argparse
import os

import numpy as np
import pandas as pd
import torch

from model import NBeats


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(path, device):

    print("Loading checkpoint:", path)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    # Check that this is an N-BEATS checkpoint
    required_keys = [
        "model_state_dict",
        "model_config",
        "series_ids",
        "statistics",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(
                f"Checkpoint is missing required key: {key}"
            )

    config = checkpoint["model_config"]

    print("Checkpoint configuration:")
    print("  backcast_length:", config["backcast_length"])
    print("  forecast_length:", config["forecast_length"])
    print("  hidden_size:", config["hidden_size"])
    print("  theta_size:", config["theta_size"])
    print("  n_blocks:", config["n_blocks"])
    print("  n_layers:", config["n_layers"])

    # Recreate exactly the same N-BEATS architecture
    # that was used during training.
    model = NBeats(
        backcast_length=config["backcast_length"],
        forecast_length=config["forecast_length"],
        hidden_size=config["hidden_size"],
        theta_size=config["theta_size"],
        n_blocks=config["n_blocks"],
        n_layers=config["n_layers"],
    )

    # IMPORTANT:
    # The checkpoint is a dictionary containing metadata.
    # We only load the actual model weights.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return model, checkpoint


# ============================================================
# Prediction
# ============================================================

def make_prediction(
    model,
    history,
    mean,
    std,
    device,
):
    """
    Use the final backcast_length observations
    to predict forecast_length future observations.
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

    # --------------------------------------------------------
    # Take the most recent history
    # --------------------------------------------------------

    x = history[-backcast_length:]

    # --------------------------------------------------------
    # Normalize using the statistics saved during training
    # --------------------------------------------------------

    x = (
        x - mean
    ) / std

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # Shape:
    #
    # [backcast_length]
    #
    # becomes:
    #
    # [1, backcast_length]
    # --------------------------------------------------------

    x = torch.tensor(
        x,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(x)

    # Shape:
    #
    # [1, forecast_length]
    #
    # -> [forecast_length]

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Convert back to original target scale
    # --------------------------------------------------------

    prediction = (
        prediction * std
        + mean
    )

    return prediction


# ============================================================
# Main
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_path = os.path.join(
        args.input_dir,
        "train.csv",
    )

    forecast_path = os.path.join(
        args.input_dir,
        "forecast_index_validation.csv",
    )

    print()
    print("Train file:")
    print(" ", os.path.abspath(train_path))

    print("Exists:", os.path.exists(train_path))

    print()
    print("Forecast index:")
    print(" ", os.path.abspath(forecast_path))

    print("Exists:", os.path.exists(forecast_path))

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Could not find train.csv at:\n"
            f"{os.path.abspath(train_path)}"
        )

    if not os.path.exists(forecast_path):
        raise FileNotFoundError(
            f"Could not find forecast_index_validation.csv at:\n"
            f"{os.path.abspath(forecast_path)}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading train.csv...")

    train = pd.read_csv(
        train_path
    )

    print(
        f"Loaded {len(train):,} training rows."
    )

    print()
    print("Loading forecast_index_validation.csv...")

    forecast_index = pd.read_csv(
        forecast_path
    )

    print(
        f"Loaded {len(forecast_index):,} forecast rows."
    )

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    train["timestamp"] = pd.to_datetime(
        train["timestamp"]
    )

    forecast_index["timestamp"] = pd.to_datetime(
        forecast_index["timestamp"]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    train = train.sort_values(
        ["series_id", "timestamp"]
    ).reset_index(drop=True)

    forecast_index = forecast_index.sort_values(
        ["series_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model, checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    statistics = checkpoint["statistics"]

    # --------------------------------------------------------
    # Check number of series
    # --------------------------------------------------------

    train_series = set(
        train["series_id"].unique()
    )

    forecast_series = set(
        forecast_index["series_id"].unique()
    )

    checkpoint_series = set(
        checkpoint["series_ids"]
    )

    print()
    print("Train series:", len(train_series))
    print("Forecast series:", len(forecast_series))
    print("Checkpoint series:", len(checkpoint_series))

    if forecast_series - train_series:
        raise ValueError(
            "Some forecast series are missing from train.csv: "
            + str(sorted(forecast_series - train_series))
        )

    if forecast_series - checkpoint_series:
        raise ValueError(
            "Some forecast series are missing from checkpoint: "
            + str(sorted(forecast_series - checkpoint_series))
        )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = []

    print()
    print("Generating predictions...")

    for series_number, (
        series_id,
        forecast_group,
    ) in enumerate(
        forecast_index.groupby(
            "series_id",
            sort=False,
        ),
        start=1,
    ):

        forecast_group = forecast_group.sort_values(
            "timestamp"
        )

        # ----------------------------------------------------
        # Get historical data for this series
        # ----------------------------------------------------

        history_df = train[
            train["series_id"] == series_id
        ].sort_values(
            "timestamp"
        )

        if history_df.empty:
            raise ValueError(
                f"No training history found for {series_id}"
            )

        history = (
            history_df["target"]
            .astype(float)
            .to_numpy()
        )

        # ----------------------------------------------------
        # Check target values
        # ----------------------------------------------------

        if np.isnan(history).any():
            raise ValueError(
                f"Target contains NaN values in {series_id}"
            )

        # ----------------------------------------------------
        # Check that we have enough history
        # ----------------------------------------------------

        if len(history) < model.backcast_length:
            raise ValueError(
                f"{series_id}: only {len(history)} "
                f"historical observations available, "
                f"but model requires "
                f"{model.backcast_length}"
            )

        # ----------------------------------------------------
        # Check statistics
        # ----------------------------------------------------

        if series_id not in statistics:
            raise ValueError(
                f"Series {series_id} "
                f"not present in checkpoint statistics."
            )

        mean = statistics[series_id]["mean"]
        std = statistics[series_id]["std"]

        # ----------------------------------------------------
        # Check forecast horizon
        # ----------------------------------------------------

        horizon = len(forecast_group)

        if horizon != model.forecast_length:
            raise ValueError(
                f"{series_id}: expected "
                f"{model.forecast_length} forecast rows, "
                f"got {horizon}"
            )

        # ----------------------------------------------------
        # Check timestamp continuity
        # ----------------------------------------------------

        last_train_timestamp = (
            history_df["timestamp"].max()
        )

        first_forecast_timestamp = (
            forecast_group["timestamp"].min()
        )

        expected_first_timestamp = (
            last_train_timestamp
            + pd.Timedelta(hours=1)
        )

        if first_forecast_timestamp != expected_first_timestamp:

            raise ValueError(
                f"{series_id}: forecast does not "
                f"immediately follow training history.\n"
                f"Last training timestamp: "
                f"{last_train_timestamp}\n"
                f"First forecast timestamp: "
                f"{first_forecast_timestamp}\n"
                f"Expected: "
                f"{expected_first_timestamp}"
            )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = make_prediction(
            model=model,
            history=history,
            mean=mean,
            std=std,
            device=device,
        )

        # ----------------------------------------------------
        # Check prediction
        # ----------------------------------------------------

        if len(prediction) != horizon:
            raise RuntimeError(
                f"{series_id}: model returned "
                f"{len(prediction)} predictions, "
                f"expected {horizon}"
            )

        if not np.isfinite(prediction).all():
            raise RuntimeError(
                f"{series_id}: prediction contains "
                f"NaN or infinite values."
            )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

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

        print(
            f"[{series_number:03d}/{len(forecast_series):03d}] "
            f"{series_id}: "
            f"{len(prediction)} predictions"
        )

    # ========================================================
    # Create result
    # ========================================================

    result = pd.DataFrame(
        predictions
    )

    # --------------------------------------------------------
    # Restore EXACT forecast_index ordering
    # --------------------------------------------------------

    result = forecast_index[
        ["series_id", "timestamp"]
    ].merge(
        result,
        on=["series_id", "timestamp"],
        how="left",
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if len(result) != len(forecast_index):
        raise RuntimeError(
            f"Output row count mismatch. "
            f"Expected {len(forecast_index):,}, "
            f"got {len(result):,}"
        )

    if result["target"].isna().any():
        raise RuntimeError(
            "Some predictions are missing."
        )

    if not np.isfinite(
        result["target"].to_numpy()
    ).all():
        raise RuntimeError(
            "Predictions contain NaN or infinite values."
        )

    # ========================================================
    # Save
    # ========================================================

    output_directory = os.path.dirname(
        os.path.abspath(args.output_file)
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    result.to_csv(
        args.output_file,
        index=False,
    )

    # ========================================================
    # Final information
    # ========================================================

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    print(
        "Output:",
        os.path.abspath(args.output_file)
    )

    print(
        "Rows:",
        f"{len(result):,}"
    )

    print(
        "Series:",
        result["series_id"].nunique()
    )

    print(
        "Forecast horizon:",
        model.forecast_length
    )

    print(
        "Prediction min:",
        float(result["target"].min())
    )

    print(
        "Prediction max:",
        float(result["target"].max())
    )

    print()
    print("First 5 predictions:")
    print(
        result.head().to_string(index=False)
    )

    print()
    print("Last 5 predictions:")
    print(
        result.tail().to_string(index=False)
    )

    print("=" * 60)


# ============================================================
# Command line
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing train.csv and forecast_index_validation.csv",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Path where predictions.csv will be written",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained N-BEATS checkpoint",
    )

    args = parser.parse_args()

    main(args)