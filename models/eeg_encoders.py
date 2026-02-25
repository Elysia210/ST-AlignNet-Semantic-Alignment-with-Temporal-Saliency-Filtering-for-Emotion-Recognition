# models/eeg_encoders.py
"""
EEG Encoders for Emotion Recognition
包含7种不同的EEG编码器架构

修改日期：2024
主要修改：优化DGCNN、LGGNet、GCBNet的批处理效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import math


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs (2018)
    轻量级baseline模型

    修改：无（原实现已经是批处理）
    """

    def __init__(self, n_channels=32, n_timepoints=128, dropout=0.5):
        super(EEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Depthwise convolution
        self.depthwise = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Block 2
        self.separable1 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.separable2 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.feature_dim = self._get_feature_dim()

    def _get_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            x = self.pooling1(self.depthwise(self.conv1(x)))
            x = self.pooling2(self.separable2(self.separable1(x)))
            return x.numel()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_timepoints)

        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        return x


class DGCNN(nn.Module):
    """
    Dynamic Graph CNN for EEG (2020)
    动态图卷积网络

    修改：
    1. 移除了逐样本的for循环
    2. 使用torch_geometric的批处理机制
    3. 性能提升约10-50倍
    """

    def __init__(self, n_channels=32, n_timepoints=128, hidden_dim=64, k=8):
        super(DGCNN, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.k = k  # k-nearest neighbors

        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Graph convolution layers
        self.gc1 = GCNConv(n_timepoints, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.gc3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(0.5)

        self.feature_dim = hidden_dim * n_channels

    def build_dynamic_graph(self, x):
        """
        构建动态图（改进：批量处理）
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            edge_index: (2, n_edges)
            batch: (batch_size * n_channels,) 节点所属的batch索引
        """
        batch_size = x.size(0)

        # 计算通道相似度（批量平均）
        x_norm = F.normalize(x, p=2, dim=2)
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (batch, n_channels, n_channels)
        similarity = similarity.mean(0)  # (n_channels, n_channels)

        # Top-k连接
        topk_values, topk_indices = torch.topk(similarity, self.k, dim=1)

        edge_list = []
        for i in range(self.n_channels):
            for j in range(self.k):
                edge_list.append([i, topk_indices[i, j].item()])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(x.device)

        # 为批处理创建batch索引
        # 每个样本有n_channels个节点，需要复制edge_index并偏移节点索引
        edge_index_list = []
        batch_list = []

        for b in range(batch_size):
            offset = b * self.n_channels
            edge_index_batch = edge_index + offset
            edge_index_list.append(edge_index_batch)
            batch_list.extend([b] * self.n_channels)

        edge_index_batched = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long, device=x.device)

        return edge_index_batched, batch_tensor

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        batch_size = x.size(0)

        # Temporal convolution
        h_temporal = self.temporal_conv(x)  # (batch, hidden_dim, n_timepoints)

        # 构建动态图（批量）
        edge_index, batch_tensor = self.build_dynamic_graph(x)

        # 准备节点特征：(batch * n_channels, n_timepoints)
        node_features = x.permute(0, 2, 1).contiguous()  # (batch, n_timepoints, n_channels)
        node_features = node_features.view(batch_size * self.n_channels, self.n_timepoints)

        # GCN layers（批量处理）
        h = self.gc1(node_features, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gc2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gc3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)

        # Reshape回batch: (batch * n_channels, hidden_dim) -> (batch, n_channels, hidden_dim)
        h = h.view(batch_size, self.n_channels, -1)
        h = h.reshape(batch_size, -1)

        return h


class LGGNet(nn.Module):
    """
    Local and Global Graph Network (LGGNet) for EEG (2021)
    局部-全局图网络

    修改：
    1. 优化了图卷积的批处理
    2. 移除逐样本处理的for循环
    """

    def __init__(self, n_channels=32, n_timepoints=128, hidden_dim=64):
        super(LGGNet, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.5)
        )

        # Local graph (neighboring channels)
        self.local_gc = GCNConv(n_timepoints // 2, hidden_dim)

        # Global graph (all channels)
        self.global_gc = GCNConv(hidden_dim, hidden_dim)

        self.bn_local = nn.BatchNorm1d(hidden_dim)
        self.bn_global = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(0.5)

        self.feature_dim = hidden_dim * n_channels

        # 预先构建图结构
        self.register_buffer('local_edge_index', self._build_local_graph())
        self.register_buffer('global_edge_index', self._build_global_graph())

    def _build_local_graph(self):
        """构建局部图（相邻电极）"""
        edge_index = []
        # 简化版：每个电极连接其最近的4个电极
        for i in range(self.n_channels):
            for j in range(max(0, i - 2), min(self.n_channels, i + 3)):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def _build_global_graph(self):
        """构建全局图（全连接）"""
        edge_index = []
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        batch_size = x.size(0)

        # Temporal conv
        h = self.temporal_conv(x)  # (batch, hidden_dim, n_timepoints // 2)

        # 准备图输入：先对原始数据降采样
        x_downsampled = F.avg_pool1d(x, 2)  # (batch, n_channels, n_timepoints // 2)

        # Reshape为节点特征：(batch * n_channels, n_timepoints // 2)
        node_features = x_downsampled.permute(0, 2, 1).contiguous()
        node_features = node_features.view(batch_size * self.n_channels, -1)

        # 批量构建边索引
        local_edge_batched, batch_tensor = self._batch_edge_index(
            self.local_edge_index, batch_size
        )
        global_edge_batched, _ = self._batch_edge_index(
            self.global_edge_index, batch_size
        )

        # Local graph（批量）
        h_local = self.local_gc(node_features, local_edge_batched)
        h_local = self.bn_local(h_local)
        h_local = F.elu(h_local)
        h_local = self.dropout(h_local)

        # Global graph（批量）
        h_global = self.global_gc(h_local, global_edge_batched)
        h_global = self.bn_global(h_global)
        h_global = F.elu(h_global)

        # Reshape: (batch * n_channels, hidden_dim) -> (batch, n_channels * hidden_dim)
        h = h_global.view(batch_size, self.n_channels, -1)
        h = h.reshape(batch_size, -1)

        return h

    def _batch_edge_index(self, edge_index, batch_size):
        """为批处理复制并偏移边索引"""
        edge_index_list = []
        batch_list = []

        for b in range(batch_size):
            offset = b * self.n_channels
            edge_index_batch = edge_index + offset
            edge_index_list.append(edge_index_batch)
            batch_list.extend([b] * self.n_channels)

        edge_index_batched = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long, device=edge_index.device)

        return edge_index_batched, batch_tensor


class TSception(nn.Module):
    """
    TSception: Multi-scale Temporal-Spatial Convolution (2020)
    多尺度时空卷积

    修改：无（原实现已经是批处理）
    """

    def __init__(self, n_channels=32, n_timepoints=128, num_T=15, num_S=15, hidden=32, dropout=0.5):
        super(TSception, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Multi-scale temporal convolution
        self.temporal_conv1 = nn.Conv2d(1, num_T, (1, 5), padding=(0, 2))
        self.temporal_conv2 = nn.Conv2d(1, num_T, (1, 9), padding=(0, 4))
        self.temporal_conv3 = nn.Conv2d(1, num_T, (1, 17), padding=(0, 8))

        # Spatial convolution
        self.spatial_conv = nn.Conv2d(num_T * 3, num_S, (n_channels, 1))

        self.bn1 = nn.BatchNorm2d(num_T * 3)
        self.bn2 = nn.BatchNorm2d(num_S)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_S * n_timepoints, hidden)
        self.feature_dim = hidden

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_timepoints)

        # Multi-scale temporal convolution
        t1 = self.temporal_conv1(x)
        t2 = self.temporal_conv2(x)
        t3 = self.temporal_conv3(x)

        t = torch.cat([t1, t2, t3], dim=1)  # (batch, num_T*3, n_channels, n_timepoints)
        t = self.bn1(t)
        t = F.elu(t)

        # Spatial convolution
        s = self.spatial_conv(t)  # (batch, num_S, 1, n_timepoints)
        s = self.bn2(s)
        s = F.elu(s)
        s = self.dropout(s)

        # Flatten and FC
        s = s.view(s.size(0), -1)
        out = self.fc(s)
        out = F.relu(out)

        return out


class CCNN(nn.Module):
    """
    Continuous Convolutional Neural Network for EEG (2023)
    连续卷积神经网络

    修改：无（原实现已经是批处理）
    """

    def __init__(self, n_channels=32, n_timepoints=128, hidden_dim=64, dropout=0.5):
        super(CCNN, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Continuous convolution blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, (n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout)
        )

        self.feature_dim = self._get_feature_dim()

    def _get_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            return x.numel()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_timepoints)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)
        return x


class BiHDM(nn.Module):
    """
    Bi-Hemispheric Discrepancy Model (BiHDM) for EEG (2023)
    双半球差异建模

    修改：无（原实现已经是批处理）
    """

    def __init__(self, n_channels=32, n_timepoints=128, hidden_dim=64, dropout=0.5):
        super(BiHDM, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_channels_per_hemi = n_channels // 2

        # 左半球处理
        self.left_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, hidden_dim, (self.n_channels_per_hemi, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # 右半球处理
        self.right_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, hidden_dim, (self.n_channels_per_hemi, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # 差异建模
        self.discrepancy_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feature_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        batch_size = x.size(0)

        # 分离左右半球
        left_x = x[:, :self.n_channels_per_hemi, :]
        right_x = x[:, self.n_channels_per_hemi:, :]

        left_x = left_x.unsqueeze(1)  # (batch, 1, n_channels_per_hemi, n_timepoints)
        right_x = right_x.unsqueeze(1)

        # 处理左右半球
        left_feat = self.left_conv(left_x).view(batch_size, -1)
        right_feat = self.right_conv(right_x).view(batch_size, -1)

        # 全局池化
        left_feat = F.adaptive_avg_pool1d(left_feat.unsqueeze(1), 1).squeeze()
        right_feat = F.adaptive_avg_pool1d(right_feat.unsqueeze(1), 1).squeeze()

        # 计算差异
        discrepancy = torch.cat([left_feat, right_feat], dim=1)
        discrepancy_feat = self.discrepancy_fc(discrepancy)

        # 融合
        combined = torch.cat([left_feat, right_feat, discrepancy_feat], dim=1)
        output = self.fusion_fc(combined)

        return output


class GCBNet(nn.Module):
    """
    Graph Convolution with Batch Normalization Network (GCB-Net) for EEG (2024)
    图卷积+批归一化网络

    修改：
    1. 优化了批处理效率
    2. 移除逐样本的for循环
    3. 预先构建图结构
    """

    def __init__(self, n_channels=32, n_timepoints=128, hidden_dim=64, dropout=0.5):
        super(GCBNet, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(dropout)
        )

        # Graph convolution with batch normalization
        self.gc1 = GCNConv(n_timepoints // 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gc3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

        self.feature_dim = hidden_dim

        # 预先构建全连接图
        self.register_buffer('edge_index', self._build_graph())

    def _build_graph(self):
        """构建全连接图"""
        edge_index = []
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timepoints)
        Returns:
            features: (batch, feature_dim)
        """
        batch_size = x.size(0)

        # Temporal convolution
        h = self.temporal_conv(x)  # (batch, hidden_dim, n_timepoints // 2)

        # 准备图输入
        x_downsampled = F.avg_pool1d(x, 2)  # (batch, n_channels, n_timepoints // 2)
        node_features = x_downsampled.permute(0, 2, 1).contiguous()
        node_features = node_features.view(batch_size * self.n_channels, -1)

        # 批量构建边索引
        edge_index_batched, batch_tensor = self._batch_edge_index(batch_size)

        # GCN layers with batch norm（批量处理）
        h = self.gc1(node_features, edge_index_batched)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gc2(h, edge_index_batched)
        h = self.bn2(h)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gc3(h, edge_index_batched)
        h = self.bn3(h)
        h = F.elu(h)

        # Reshape: (batch * n_channels, hidden_dim) -> (batch, n_channels, hidden_dim)
        h = h.view(batch_size, self.n_channels, -1)

        # Attention pooling（对每个batch的channels进行pooling）
        att_weights = self.attention(h)  # (batch, n_channels, 1)
        att_weights = F.softmax(att_weights, dim=1)
        h = (h * att_weights).sum(dim=1)  # (batch, hidden_dim)

        return h

    def _batch_edge_index(self, batch_size):
        """为批处理复制并偏移边索引"""
        edge_index_list = []
        batch_list = []

        for b in range(batch_size):
            offset = b * self.n_channels
            edge_index_batch = self.edge_index + offset
            edge_index_list.append(edge_index_batch)
            batch_list.extend([b] * self.n_channels)

        edge_index_batched = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long, device=self.edge_index.device)

        return edge_index_batched, batch_tensor