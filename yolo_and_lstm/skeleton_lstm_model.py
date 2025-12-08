import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=30, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        fc_in_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        out, (h_n, c_n) = self.lstm(x)  # out: [B, L, H*dir]
        # 取最后一帧的输出（也可以 mean pooling 等）
        last_out = out[:, -1, :]        # [B, H*dir]
        logits = self.fc(last_out)      # [B, C]
        return logits
