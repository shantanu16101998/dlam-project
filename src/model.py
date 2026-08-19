from __future__ import annotations

import torch


class ForecastModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 23,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 24,
    ) -> None:
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.fc = torch.nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        return self.fc(hidden)