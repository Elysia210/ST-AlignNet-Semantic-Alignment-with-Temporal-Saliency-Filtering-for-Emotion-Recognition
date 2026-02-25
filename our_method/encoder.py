import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):  # x: [B, T, D]
        return x + self.pe[:, :x.size(1)]

class EEGEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(16 * 128, embed_dim)
        self.pe = PositionalEncoding(d_model=128)

    def forward(self, x):  # x: [B, 32, 128]
        #print("EEG shape before reshape:", x.shape)
        x = x.unsqueeze(1)  # [B, 1, 32, 128]
        x = self.conv(x)    # [B, 16, 32, 128]
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), x.size(2), -1)  # [B, 32, 16*128]
        x = self.linear(x)  # [B, T, D]= [B, 32, embed_dim]
        x = self.pe(x) # 加上位置编码
        return x

class FacialEncoder(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=128):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, embed_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x, tgt_len=None):  # x: [B, T, 2048]
        x = x.transpose(1, 2)  # [B, 2048, T]
        x = self.conv1d(x)     # [B, 128, T]
        x = self.bn(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # [B, T, 128]

        #  自动对齐 EEG 时间步（如果提供）
        if tgt_len is not None and x.size(1) != tgt_len:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), tgt_len).transpose(1, 2)

        return x  # [B, tgt_len, 128]
