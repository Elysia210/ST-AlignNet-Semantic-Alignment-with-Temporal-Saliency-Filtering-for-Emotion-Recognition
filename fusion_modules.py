# fusion_modules.py
"""
多模态融合模块 - 8种融合方法
支持EEG和Facial特征的融合

方法分类:
1. 基础操作类 (F1-F3): Concatenation, Element-wise Sum, Element-wise Product
2. 深度融合类 (F4-F6): Gated, MLP, Bilinear Pooling
3. 注意力类 (F7-F8): Cross-Attention, Co-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseFusion(nn.Module):
    """融合基类"""
    def __init__(self, dim_eeg, dim_facial, output_dim=None):
        super().__init__()
        self.dim_eeg = dim_eeg
        self.dim_facial = dim_facial
        self.output_dim = output_dim or max(dim_eeg, dim_facial)

    def forward(self, feat_eeg, feat_facial):
        raise NotImplementedError


# ==================== 基础操作类 ====================

class ConcatFusion(BaseFusion):
    """
    F1: Concatenation Fusion
    论文: Multimodal Learning with Deep Boltzmann Machines (Srivastava et al., JMLR 2014)
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None):
        super().__init__(dim_eeg, dim_facial, output_dim)
        self.fc = nn.Linear(dim_eeg + dim_facial, self.output_dim)

    def forward(self, feat_eeg, feat_facial):
        concat = torch.cat([feat_eeg, feat_facial], dim=-1)
        return self.fc(concat)


class SumFusion(BaseFusion):
    """
    F2: Element-wise Sum Fusion (with learnable weights)
    论文: Deep Residual Learning (He et al., CVPR 2016) - 残差连接启发
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None):
        super().__init__(dim_eeg, dim_facial, output_dim)
        assert dim_eeg == dim_facial, "SumFusion requires same dimensions"
        
        # 可学习权重 (每个维度独立)
        self.alpha = nn.Parameter(torch.ones(dim_eeg))
        self.beta = nn.Parameter(torch.ones(dim_facial))
        
        if output_dim and output_dim != dim_eeg:
            self.proj = nn.Linear(dim_eeg, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, feat_eeg, feat_facial):
        fused = self.alpha * feat_eeg + self.beta * feat_facial
        return self.proj(fused)


class ProductFusion(BaseFusion):
    """
    F3: Element-wise Product Fusion (Hadamard Product)
    论文: Multimodal Compact Bilinear Pooling (Fukui et al., EMNLP 2016)
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None):
        super().__init__(dim_eeg, dim_facial, output_dim)
        assert dim_eeg == dim_facial, "ProductFusion requires same dimensions"
        
        if output_dim and output_dim != dim_eeg:
            self.fc = nn.Linear(dim_eeg, output_dim)
        else:
            self.fc = nn.Identity()

    def forward(self, feat_eeg, feat_facial):
        product = feat_eeg * feat_facial
        return self.fc(product)


# ==================== 深度融合类 ====================

class GatedFusion(BaseFusion):
    """
    F4: Gated Fusion
    论文: Gated Multimodal Units (Arevalo et al., ACL 2017)
    动态学习每个模态的重要性
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None):
        super().__init__(dim_eeg, dim_facial, output_dim)
        
        # 维度对齐
        if dim_eeg != dim_facial:
            self.proj_eeg = nn.Linear(dim_eeg, self.output_dim)
            self.proj_facial = nn.Linear(dim_facial, self.output_dim)
        else:
            self.proj_eeg = nn.Identity()
            self.proj_facial = nn.Identity()
            if not output_dim:
                self.output_dim = dim_eeg
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, feat_eeg, feat_facial):
        # 投影到相同维度
        f_e = self.proj_eeg(feat_eeg)
        f_f = self.proj_facial(feat_facial)
        
        # 计算门控值
        gate_input = torch.cat([f_e, f_f], dim=-1)
        z = self.gate(gate_input)
        
        # 门控融合
        fused = z * f_e + (1 - z) * f_f
        return fused


class MLPFusion(BaseFusion):
    """
    F5: MLP Fusion
    论文: Neural Module Networks (Andreas et al., CVPR 2016)
    多层非线性变换学习复杂融合函数
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None, hidden_dim=256, dropout=0.5):
        super().__init__(dim_eeg, dim_facial, output_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim_eeg + dim_facial, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, feat_eeg, feat_facial):
        concat = torch.cat([feat_eeg, feat_facial], dim=-1)
        return self.fusion_mlp(concat)


class BilinearPooling(BaseFusion):
    """
    F6: Bilinear Pooling
    论文: Multimodal Compact Bilinear Pooling (Fukui et al., EMNLP 2016)
    使用低秩分解减少参数量
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=256):
        super().__init__(dim_eeg, dim_facial, output_dim)
        
        # 低秩投影
        rank = min(output_dim, dim_eeg, dim_facial)
        self.proj_eeg = nn.Linear(dim_eeg, rank)
        self.proj_facial = nn.Linear(dim_facial, rank)
        
        if output_dim != rank:
            self.out_proj = nn.Linear(rank, output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, feat_eeg, feat_facial):
        # 投影到低秩空间
        f_e = self.proj_eeg(feat_eeg)  # (B, rank)
        f_f = self.proj_facial(feat_facial)  # (B, rank)
        
        # 双线性池化 (简化版: element-wise product)
        bilinear = f_e * f_f
        
        return self.out_proj(bilinear)


# ==================== 注意力机制类 ====================

class CrossAttentionFusion(BaseFusion):
    """
    F7: Cross-Attention Fusion
    论文: Attention is All You Need (Vaswani et al., NeurIPS 2017)
    一个模态关注另一个模态
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None, num_heads=8, dropout=0.1):
        super().__init__(dim_eeg, dim_facial, output_dim)
        
        # 维度对齐
        if dim_eeg != dim_facial:
            common_dim = min(dim_eeg, dim_facial)
            self.proj_eeg = nn.Linear(dim_eeg, common_dim)
            self.proj_facial = nn.Linear(dim_facial, common_dim)
        else:
            common_dim = dim_eeg
            self.proj_eeg = nn.Identity()
            self.proj_facial = nn.Identity()
        
        # 多头注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(common_dim)
        
        if output_dim and output_dim != common_dim:
            self.out_proj = nn.Linear(common_dim, output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, feat_eeg, feat_facial):
        # 投影
        f_e = self.proj_eeg(feat_eeg)  # (B, dim)
        f_f = self.proj_facial(feat_facial)  # (B, dim)
        
        # 添加序列维度
        q = f_e.unsqueeze(1)  # (B, 1, dim)
        k = v = f_f.unsqueeze(1)  # (B, 1, dim)
        
        # Cross-attention: EEG queries Facial
        attn_out, _ = self.cross_attn(q, k, v)  # (B, 1, dim)
        
        # 残差连接和归一化
        fused = self.norm(attn_out.squeeze(1) + f_e)
        
        return self.out_proj(fused)


class CoAttentionFusion(BaseFusion):
    """
    F8: Co-Attention Fusion
    论文: Hierarchical Question-Image Co-Attention (Lu et al., NeurIPS 2016)
    两个模态互相关注
    """
    def __init__(self, dim_eeg, dim_facial, output_dim=None, hidden_dim=128, dropout=0.1):
        super().__init__(dim_eeg, dim_facial, output_dim)
        
        # 维度对齐
        if dim_eeg != dim_facial:
            common_dim = min(dim_eeg, dim_facial)
            self.proj_eeg = nn.Linear(dim_eeg, common_dim)
            self.proj_facial = nn.Linear(dim_facial, common_dim)
        else:
            common_dim = dim_eeg
            self.proj_eeg = nn.Identity()
            self.proj_facial = nn.Identity()
        
        # 相似度计算
        self.W_sim = nn.Linear(common_dim, hidden_dim)
        
        # 双向注意力投影
        self.attn_eeg = nn.Linear(common_dim, common_dim)
        self.attn_facial = nn.Linear(common_dim, common_dim)
        
        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(common_dim * 2, output_dim or common_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, feat_eeg, feat_facial):
        # 投影
        f_e = self.proj_eeg(feat_eeg)  # (B, dim)
        f_f = self.proj_facial(feat_facial)  # (B, dim)
        
        # 计算相似度
        h_e = torch.tanh(self.W_sim(f_e))
        h_f = torch.tanh(self.W_sim(f_f))
        sim = (h_e * h_f).sum(dim=-1, keepdim=True)  # (B, 1)
        
        # 注意力权重
        alpha = torch.sigmoid(sim)
        
        # EEG关注Facial
        attended_e = alpha * self.attn_eeg(f_f) + (1 - alpha) * f_e
        
        # Facial关注EEG
        attended_f = (1 - alpha) * self.attn_facial(f_e) + alpha * f_f
        
        # 拼接融合
        fused = self.fusion_fc(torch.cat([attended_e, attended_f], dim=-1))
        
        return fused


# ==================== 工厂函数 ====================

def create_fusion_module(fusion_type, dim_eeg, dim_facial, output_dim=None, **kwargs):
    """
    工厂函数：根据类型创建融合模块
    
    Args:
        fusion_type: str, 融合类型
            'concat' / 'f1' - Concatenation
            'sum' / 'f2' - Element-wise Sum
            'product' / 'f3' - Element-wise Product
            'gated' / 'f4' - Gated Fusion
            'mlp' / 'f5' - MLP Fusion
            'bilinear' / 'f6' - Bilinear Pooling
            'cross_attn' / 'f7' - Cross-Attention
            'co_attn' / 'f8' - Co-Attention
        dim_eeg: int, EEG特征维度
        dim_facial: int, Facial特征维度
        output_dim: int, 输出维度
        **kwargs: 其他参数
    
    Returns:
        fusion_module: BaseFusion的子类实例
    """
    fusion_map = {
        'concat': ConcatFusion,
        'f1': ConcatFusion,
        'sum': SumFusion,
        'f2': SumFusion,
        'product': ProductFusion,
        'f3': ProductFusion,
        'gated': GatedFusion,
        'f4': GatedFusion,
        'mlp': MLPFusion,
        'f5': MLPFusion,
        'bilinear': BilinearPooling,
        'f6': BilinearPooling,
        'cross_attn': CrossAttentionFusion,
        'f7': CrossAttentionFusion,
        'co_attn': CoAttentionFusion,
        'f8': CoAttentionFusion,
    }
    
    fusion_type_lower = fusion_type.lower()
    if fusion_type_lower not in fusion_map:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                        f"Available: {list(fusion_map.keys())}")
    
    return fusion_map[fusion_type_lower](dim_eeg, dim_facial, output_dim, **kwargs)
