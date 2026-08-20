import argparse
import os

import numpy as np
import pandas as pd
import torch


from model import NBeats


def load_checkpoint(path, device):

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    Config = checkpoint["model_config"]

    model = NBeats(
        backcast_length=Config["backcast_length"],
        forecast_length=Config["forecast_length"],
        hidden_size=Config["hidden_size"],
        theta_size=Config["theta_size"],
        number_of_blocks=Config["n_blocks"],
        number_of_layers=Config["n_layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict(
    model,
    historical_data,
    mean,
    std_dev,
    device,
):
    history = np.asarray(historical_data, dtype=np.float32)

    # Getting last part of data
    unnormalized_data = history[-model.backcast_length :]

    normalized_data = (unnormalized_data - mean) / std_dev

    input_to_model = torch.tensor(
        normalized_data, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_to_model)

    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = prediction * std_dev + mean

    return prediction


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Training df
    train_csv_path = os.path.join(args.input_dir, "train.csv")
    train_dataframe = pd.read_csv(train_csv_path, parse_dates=["timestamp"])
    train_dataframe = train_dataframe.sort_values(
        ["series_id", "timestamp"]
    ).reset_index(drop=True)
    train_series = set(train_dataframe["series_id"].unique())

    # Forecast df
    forecast_csv_pth = os.path.join(args.input_dir, "forecast_index_validation.csv")
    forecast_dataframe = pd.read_csv(forecast_csv_pth, parse_dates=["timestamp"])
    forecast_dataframe = forecast_dataframe.sort_values(
        ["series_id", "timestamp"]
    ).reset_index(drop=True)
    forecast_series = set(forecast_dataframe["series_id"].unique())
    
    # Checkpoint loaded
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_series = set(checkpoint["series_ids"])

    statistics = checkpoint["statistics"]

    predictions = []
    series_no = 0

    for series_id, forecast_group in forecast_dataframe.groupby("series_id"):

        forecast_group = forecast_group.sort_values("timestamp")

        history_df = train_dataframe[train_dataframe["series_id"] == series_id]
        history_df = history_df.sort_values("timestamp")

        history = history_df["target"].astype(float).to_numpy()

        prediction = predict(
            model=model,
            historical_data=history,
            mean=statistics[series_id]["mean"],
            std_dev=statistics[series_id]["std"],
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
                    "prediction": float(value),
                }
            )

        series_no += 1
        print(f"predicting this series: {series_no}")
        

    result = pd.DataFrame(predictions)

    result = forecast_dataframe[["series_id", "timestamp"]].merge(
        result,
        on=["series_id", "timestamp"],
        how="left",
    )

    output_directory = os.path.dirname(os.path.abspath(args.output_file))

    os.makedirs(
        output_directory,
        exist_ok=True,
    )


    result.to_csv(
        args.output_file,
        index=False,
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