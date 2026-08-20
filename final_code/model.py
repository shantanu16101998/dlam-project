import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """
    Basic N-BEATS block.

    Takes a backcast window and produces:
        - backcast: reconstruction of the input
        - forecast: prediction for the requested horizon
    """

    def __init__(
        self,
        input_size,
        theta_size,
        hidden_size,
        n_layers,
        forecast_size,
    ):
        super().__init__()

        layers = []

        in_features = input_size

        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        self.fc = nn.Sequential(*layers)

        self.theta = nn.Linear(hidden_size, theta_size)

        # Split theta into backcast and forecast coefficients.
        self.backcast_linear = nn.Linear(
            theta_size,
            input_size,
        )

        self.forecast_linear = nn.Linear(
            theta_size,
            forecast_size,
        )

    def forward(self, x):
        h = self.fc(x)
        theta = self.theta(h)

        backcast = self.backcast_linear(theta)
        forecast = self.forecast_linear(theta)

        return backcast, forecast


class NBeats(nn.Module):
    """
    Generic N-BEATS model.

    Input:
        [batch, backcast_length]

    Output:
        [batch, forecast_length]
    """

    def __init__(
        self,
        backcast_length=672,
        forecast_length=336,
        hidden_size=256,
        theta_size=128,
        n_blocks=6,
        n_layers=4,
    ):
        super().__init__()

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(
                    input_size=backcast_length,
                    theta_size=theta_size,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    forecast_size=forecast_length,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x):
        residual = x
        forecast = torch.zeros(
            x.size(0),
            self.forecast_length,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            backcast, block_forecast = block(residual)

            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast

