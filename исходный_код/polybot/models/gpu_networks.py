"""GPU-модели проекта. Импорт файла не требует установленного PyTorch."""

from __future__ import annotations


def build_network(name: str, feature_count: int, hidden_size: int = 96):
    """Создаёт multi-task сеть: направление, utility, fill и преимущество выхода."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch ещё не установлен. Запустите подготовить_gpu_компьютер.py --execute") from exc

    class MultiTaskHead(nn.Module):
        def __init__(self, encoder: nn.Module, output_size: int):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(nn.LayerNorm(output_size), nn.Dropout(0.15))
            self.direction = nn.Linear(output_size, 1)
            self.utility = nn.Linear(output_size, 3)  # HOLD / UP / DOWN
            self.fill = nn.Linear(output_size, 1)
            self.exit_advantage = nn.Linear(output_size, 1)

        def forward(self, values):
            encoded = self.encoder(values)
            if isinstance(encoded, tuple):
                encoded = encoded[0][:, -1]
            elif encoded.ndim == 3:
                encoded = encoded[:, -1]
            encoded = self.head(encoded)
            return {
                "direction_logit": self.direction(encoded).squeeze(-1),
                "action_utility": self.utility(encoded),
                "fill_logit": self.fill(encoded).squeeze(-1),
                "exit_advantage": self.exit_advantage(encoded).squeeze(-1),
            }

    normalized = name.lower().replace("-", "_")
    if normalized == "gru":
        encoder = nn.GRU(feature_count, hidden_size, batch_first=True, num_layers=2, dropout=0.15)
        return MultiTaskHead(encoder, hidden_size)
    if normalized == "tcn":
        encoder = nn.Sequential(
            nn.Conv1d(feature_count, hidden_size, 3, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=8, dilation=4), nn.GELU(),
        )

        class TCNAdapter(nn.Module):
            def __init__(self, layers):
                super().__init__(); self.layers = layers
            def forward(self, values):
                return self.layers(values.transpose(1, 2)).transpose(1, 2)[:, : values.shape[1]]
        return MultiTaskHead(TCNAdapter(encoder), hidden_size)
    if normalized in {"tiny_transformer", "transformer"}:
        projection = nn.Linear(feature_count, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 4, 0.15, batch_first=True, norm_first=True)

        class TransformerAdapter(nn.Module):
            def __init__(self):
                super().__init__(); self.projection = projection; self.encoder = nn.TransformerEncoder(layer, 3)
            def forward(self, values):
                return self.encoder(self.projection(values))
        return MultiTaskHead(TransformerAdapter(), hidden_size)
    raise ValueError(f"Неизвестная архитектура: {name}")

