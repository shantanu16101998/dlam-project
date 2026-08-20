import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):

    def __init__(
        self,
        input_size,
        theta_size,
        hidden_size,
        number_of_layers,
        forecast_size,
    ):
        super().__init__()

        layers = []

        # first is input -> hidden then hidden -> hidden
        current_input_feature_size = input_size

        for _ in range(number_of_layers):
            layers.append(nn.Linear(current_input_feature_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_feature_size = hidden_size

        self.fc = nn.Sequential(*layers)

        self.theta = nn.Linear(hidden_size, theta_size)

        self.backcast = nn.Linear(
            theta_size,
            input_size,
        )

        self.forecast = nn.Linear(
            theta_size,
            forecast_size,
        )

    def forward(self, x):
        h = self.fc(x)
        theta = self.theta(h)

        backcast = self.backcast(theta)
        forecast = self.forecast(theta)

        return backcast, forecast


class NBeats(nn.Module):

    def __init__(
        self,
        backcast_length,
        forecast_length,
        hidden_size,
        theta_size,
        number_of_blocks,
        number_of_layers,
    ):
        super().__init__()

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.blocks = nn.ModuleList()


        for i in range(number_of_blocks):
            block = NBeatsBlock(
                input_size=backcast_length,
                theta_size=theta_size,
                hidden_size=hidden_size,
                number_of_layers=number_of_layers,
                forecast_size=forecast_length,
            )

            self.blocks.append(block)

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
