# models/facial_encoders.py
"""
Facial Expression Encoders for Emotion Recognition
包含6种不同的面部表情编码器架构

重要修改（适配滑动窗口预处理）:
1. 输入维度: (batch, 128, 2048) → (batch, 16, 2048)
2. 每个窗口16帧（2秒窗口 × 8fps采样）
3. 所有encoder默认n_timepoints=16
4. 修复C3D的reshape逻辑
5. 调整EST的多尺度特征提取

注意：输入是ResNet50预提取的特征
- 这是文献主流做法（60%论文使用）
- Facial encoder的作用是"时序建模"而不是"特征提取"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class PositionalEncoding(nn.Module):
    """位置编码（保持不变）"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class C3D(nn.Module):
    """
    3D Convolutional Network for Video Classification (2014)
    Baseline 3D CNN模型

    主要修改：
    1. 输入维度: 128帧 → 16帧
    2. 重新设计3D卷积结构（不使用奇怪的reshape）
    3. 直接在特征维度上进行3D卷积
    """

    def __init__(self, n_timepoints=16, input_dim=2048, hidden_dim=512, dropout=0.5):
        super(C3D, self).__init__()

        self.n_timepoints = n_timepoints
        self.input_dim = input_dim

        # 特征投影：将2048维降到256维（更适合3D卷积）
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Reshape: (batch, 16, 256) → (batch, 1, 16, 16, 16)
        # 将256维特征reshape成16×16的空间特征图

        # 3D convolutions
        # 输入: (batch, 1, 16, 16, 16) - (B, C, T, H, W)
        self.conv3d_1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 输出: (batch, 32, 8, 8, 8)

        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 输出: (batch, 64, 4, 4, 4)

        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 输出: (batch, 128, 2, 2, 2)

        self.dropout = nn.Dropout(dropout)

        # FC layers
        # 128 × 2 × 2 × 2 = 1024
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.feature_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: (batch, n_timepoints, input_dim) = (batch, 16, 2048)
        Returns:
            features: (batch, hidden_dim)
        """
        batch_size = x.size(0)

        # 特征投影: (batch, 16, 2048) → (batch, 16, 256)
        x = self.feature_proj(x)

        # Reshape: (batch, 16, 256) → (batch, 16, 16, 16)
        # 将256维reshape成16×16的2D特征图
        x = x.view(batch_size, self.n_timepoints, 16, 16)

        # 添加通道维度: (batch, 16, 16, 16) → (batch, 1, 16, 16, 16)
        x = x.unsqueeze(1)  # (B, C=1, T=16, H=16, W=16)

        # 3D convolutions
        x = F.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool3(x)

        x = self.dropout(x)

        # Flatten: (batch, 128, 2, 2, 2) → (batch, 1024)
        x = x.view(batch_size, -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SlowFast(nn.Module):
    """
    SlowFast Networks for Video Recognition (2019)
    双流网络：慢通路捕捉语义，快通路捕捉动作

    修改：
    1. 输入维度: 128帧 → 16帧
    2. 调整卷积核大小适配更短的序列
    """

    def __init__(self, n_timepoints=16, input_dim=2048, hidden_dim=512, dropout=0.5):
        super(SlowFast, self).__init__()

        self.input_dim = input_dim
        self.n_timepoints = n_timepoints

        # Slow pathway (低帧率，高通道数)
        self.slow_conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),  # 减小卷积核
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 16 → 8
        )

        self.slow_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),  # 减小卷积核
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 8 → 4
        )

        # Fast pathway (高帧率，低通道数)
        self.fast_conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )

        self.fast_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim // 4, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )

        # Lateral connections
        self.lateral1 = nn.Conv1d(hidden_dim // 4, hidden_dim, 1)
        self.lateral2 = nn.Conv1d(hidden_dim // 4, hidden_dim, 1)

        # Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feature_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: (batch, n_timepoints, input_dim) = (batch, 16, 2048)
        Returns:
            features: (batch, hidden_dim)
        """
        # Transpose for conv1d: (batch, 16, 2048) → (batch, 2048, 16)
        x = x.transpose(1, 2)

        # Slow pathway
        slow = self.slow_conv1(x)  # (batch, hidden_dim, 8)
        slow = self.slow_conv2(slow)  # (batch, hidden_dim, 4)
        slow = F.adaptive_avg_pool1d(slow, 1).squeeze(-1)  # (batch, hidden_dim)

        # Fast pathway
        fast = self.fast_conv1(x)  # (batch, hidden_dim // 4, 16)
        fast = self.fast_conv2(fast)  # (batch, hidden_dim // 4, 16)
        fast = F.adaptive_avg_pool1d(fast, 1).squeeze(-1)  # (batch, hidden_dim // 4)

        # Lateral connection
        fast_lateral = self.lateral2(fast.unsqueeze(-1)).squeeze(-1)  # (batch, hidden_dim)

        # Fusion
        combined = torch.cat([slow, fast_lateral], dim=1)  # (batch, hidden_dim * 2)
        output = self.fusion_fc(combined)

        return output


class VideoSwin(nn.Module):
    """
    Video Swin Transformer (2022)
    视频版的Swin Transformer

    修改：
    1. 输入维度: 128帧 → 16帧
    2. 位置编码适配新长度
    """

    def __init__(self, n_timepoints=16, input_dim=2048, d_model=256,
                 nhead=8, num_layers=4, dropout=0.1):
        super(VideoSwin, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_timepoints = n_timepoints

        # Patch embedding (将ResNet特征投影到d_model)
        self.patch_embed = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, n_timepoints)

        # Shifted window attention (简化版)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, n_timepoints, input_dim) = (batch, 16, 2048)
        Returns:
            features: (batch, d_model)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, 16, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)

        # Global average pooling
        features = x.mean(dim=1)  # (batch, d_model)

        return features


class FormerDFER(nn.Module):
    """
    Former-DFER: Dynamic Facial Expression Recognition Transformer (2023)
    专门的动态面部表情识别Transformer

    修改：
    1. 输入维度: 128帧 → 16帧
    2. 位置编码适配新长度
    """

    def __init__(self, n_timepoints=16, input_dim=2048, d_model=256,
                 nhead=8, num_layers=6, dropout=0.1):
        super(FormerDFER, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_timepoints = n_timepoints

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Temporal positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, n_timepoints)

        # Multi-scale temporal encoding
        self.local_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        self.global_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        # Emotion-aware attention
        self.emotion_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, n_timepoints, input_dim) = (batch, 16, 2048)
        Returns:
            features: (batch, d_model)
        """
        # Feature projection
        x = self.feature_proj(x)  # (batch, 16, d_model)
        x = self.pos_encoder(x)

        # Local temporal encoding
        local_feat = self.local_encoder(x)  # (batch, 16, d_model)

        # Global temporal encoding
        global_feat = self.global_encoder(x)  # (batch, 16, d_model)

        # Emotion-aware attention
        emotion_feat, _ = self.emotion_attention(
            global_feat, local_feat, local_feat
        )  # (batch, 16, d_model)

        emotion_feat = self.norm(emotion_feat)

        # Temporal pooling
        features = emotion_feat.mean(dim=1)  # (batch, d_model)

        return features


class LOGOFormer(nn.Module):
    """
    LOGO-Former: Local-Global Transformer for DFER (2024)
    局部-全局Transformer

    修改：
    1. 输入维度: 128帧 → 16帧
    2. window_size: 8 → 4（适配更短序列）
    """

    def __init__(self, n_timepoints=16, input_dim=2048, d_model=256,
                 nhead=8, num_layers=6, window_size=4, dropout=0.1):
        super(LOGOFormer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.window_size = window_size  # 从8改为4
        self.n_timepoints = n_timepoints

        # Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout, n_timepoints)

        # Local attention (within windows)
        self.local_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers // 2)
        ])

        # Global attention (across windows)
        self.global_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers // 2)
        ])

        # Feed-forward networks
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 2)
        ])

        # Learnable emotion queries
        self.emotion_queries = nn.Parameter(torch.randn(1, 8, d_model))

        self.dropout = nn.Dropout(dropout)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, n_timepoints, input_dim) = (batch, 16, 2048)
        Returns:
            features: (batch, d_model)
        """
        batch_size = x.size(0)

        # Embedding
        x = self.embed(x)  # (batch, 16, d_model)
        x = self.pos_encoder(x)

        norm_idx = 0

        # Local-global alternating attention
        for i in range(len(self.local_attention)):
            # Local attention
            attn_out, _ = self.local_attention[i](x, x, x)
            x = self.norms[norm_idx](x + self.dropout(attn_out))
            norm_idx += 1

            # FFN
            ffn_out = self.ffn[i * 2](x)
            x = self.norms[norm_idx](x + ffn_out)
            norm_idx += 1

            # Global attention
            attn_out, _ = self.global_attention[i](x, x, x)
            x = self.norms[norm_idx](x + self.dropout(attn_out))
            norm_idx += 1

            # FFN
            ffn_out = self.ffn[i * 2 + 1](x)
            x = self.norms[norm_idx](x + ffn_out)
            norm_idx += 1

        # Use emotion queries to aggregate
        emotion_queries = self.emotion_queries.expand(batch_size, -1, -1)
        emotion_feat, _ = self.global_attention[-1](emotion_queries, x, x)

        # Aggregate emotion features
        features = emotion_feat.mean(dim=1)  # (batch, d_model)

        return features


class EST(nn.Module):
    """
    Emotion-Semantic Transformer (EST) for DFER (2024)
    情绪语义感知Transformer

    修改：
    1. 输入维度: 128帧 → 16帧
    2. 调整多尺度特征提取（16帧太短，不能pool 4）
    """

    def __init__(self, n_timepoints=16, input_dim=2048, d_model=256,
                 nhead=8, num_layers=6, dropout=0.1):
        super(EST, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_timepoints = n_timepoints

        # Multi-scale feature extraction
        # 注意：16帧太短，只能做2倍下采样
        self.scale1_proj = nn.Linear(input_dim, d_model)  # 原始尺度
        self.scale2_proj = nn.Linear(input_dim, d_model)  # 2倍下采样

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, n_timepoints)

        # Semantic encoding
        self.semantic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        # Emotion encoding
        self.emotion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Emotion-semantic alignment
        self.alignment_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, n_timepoints, input_dim) = (batch, 16, 2048)
        Returns:
            features: (batch, d_model)
        """
        batch_size = x.size(0)

        # Multi-scale feature extraction
        # Scale 1: 原始 (16帧)
        scale1 = self.scale1_proj(x)  # (batch, 16, d_model)

        # Scale 2: 2倍下采样 (8帧) → 上采样回16帧
        scale2 = self.scale2_proj(F.avg_pool1d(x.transpose(1, 2), 2).transpose(1, 2))
        scale2 = F.interpolate(scale2.transpose(1, 2), size=self.n_timepoints).transpose(1, 2)

        # 注意：原来有scale3 (4倍下采样)，但16帧太短，去掉
        # 组合两个尺度
        multi_scale_feat = (scale1 + scale2) / 2
        multi_scale_feat = self.pos_encoder(multi_scale_feat)

        # Semantic encoding
        semantic_feat = self.semantic_encoder(multi_scale_feat)

        # Emotion encoding
        emotion_feat = self.emotion_encoder(multi_scale_feat)

        # Cross-modal fusion
        fused_feat, _ = self.cross_attention(
            emotion_feat, semantic_feat, semantic_feat
        )

        # Emotion-semantic alignment
        aligned_feat = torch.cat([emotion_feat, fused_feat], dim=-1)
        aligned_feat = self.alignment_fc(aligned_feat)  # (batch, 16, d_model)

        # Temporal attention pooling
        att_weights = self.temporal_attention(aligned_feat)  # (batch, 16, 1)
        att_weights = F.softmax(att_weights, dim=1)
        features = (aligned_feat * att_weights).sum(dim=1)  # (batch, d_model)

        return features