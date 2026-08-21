import argparse
import numpy as np
import pandas as pd
import torch
from model import NBeats

BACKCAST_LENGTH = 168
FORECAST_LENGTH = 24


def mean_absolute_error(true_value, predicted_value):
    return np.mean(np.abs(true_value - predicted_value))


def root_mean_square_error(true_value, predicted_value):
    mean_square_error = np.mean((true_value - predicted_value) ** 2)
    return np.sqrt(mean_square_error)


def mean_absolute_percentage_error(true_value, predicted_value):
    denominator = np.maximum(np.abs(true_value), 1e-6)
    numerator = true_value - predicted_value
    
    return np.mean(np.abs( numerator / denominator)) * 100

def mean_square_error(true_value, predicted_value):
    return np.mean((true_value - predicted_value) ** 2)

def symmetric_mean_absolute_percentage_error(true_value, predicted_value):
    numerator = 2 * np.abs(predicted_value - true_value)
    denomiator = np.abs(true_value) + np.abs(predicted_value) + 1e-10

    return np.mean( numerator / denomiator) * 100

def weighted_absolute_percentage_error(true_value, predicted_value):
    numerator = np.sum(np.abs(true_value - predicted_value))
    denominator = np.sum(np.abs(true_value)) + 1e-10
    
    return ( numerator / denominator) * 100


def load_uci_from_path(path):

    dataframe = pd.read_csv(path, sep=";", low_memory=False)
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

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load( args.checkpoint, map_location=device ,weights_only=False)
    
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

    hourly_data = load_uci_from_path(args.data)
    test_train_split = int(len(hourly_data) * 0.80)

    train_dataframe = hourly_data.iloc[:test_train_split]
    test_dataframe = hourly_data.iloc[test_train_split:]

    statistics = checkpoint["statistics"]

    all_true_value = []
    all_predicted_value = []

    for column in checkpoint["series_names"]:
        
        train_values = train_dataframe[column].to_numpy(dtype=np.float32)
        test_values = test_dataframe[column].to_numpy(dtype=np.float32)

        mean = statistics[column]["mean"]
        std_dev = statistics[column]["std"]

        train_normalized = (train_values - mean) / std_dev
        test_normalized = (test_values - mean) / std_dev

        history = list(train_normalized)

        predicted_values = []
        true_values = []

        for start in range(0, len(test_normalized), FORECAST_LENGTH):
            
            true_partition = test_normalized[start : start + FORECAST_LENGTH]

            if len(true_partition) < FORECAST_LENGTH:
                break

            x = np.asarray(
                history[-BACKCAST_LENGTH:],
                dtype=np.float32,
            )
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(x_tensor)

            prediction = prediction.squeeze(0).cpu().numpy()

            predicted_values.extend(prediction)
            true_values.extend(true_partition)
            history.extend(true_partition)

        predicted_values = np.asarray(predicted_values)
        true_values = np.asarray(true_values)

        predictions_original = predicted_values * std_dev + mean
        actuals_original = true_values * std_dev + mean

        all_predicted_value.extend(predictions_original)
        all_true_value.extend(actuals_original)

    true_value = np.asarray(all_true_value, dtype=np.float32)
    predicted_value = np.asarray(all_predicted_value, dtype=np.float32)

    mae = mean_absolute_error(true_value, predicted_value)
    rmse = root_mean_square_error(true_value, predicted_value)
    mape = mean_absolute_percentage_error(true_value, predicted_value)
    mse = mean_square_error(true_value, predicted_value)
    smape = symmetric_mean_absolute_percentage_error(true_value, predicted_value)
    wape = weighted_absolute_percentage_error(true_value, predicted_value)

    print("UCI N-BEATS RESULTS")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print("Mape will be high because around 20 percent of the data contains values that are 0.")
    print(f"MAPE : {mape:.4f}%")
    print(f"MSE  : {mse:.6f}")
    print(f"SMAPE : {smape:.4f}%")
    print(f"WAPE  : {wape:.6f}")

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

    args = parser.parse_args()
    evaluate(args)
