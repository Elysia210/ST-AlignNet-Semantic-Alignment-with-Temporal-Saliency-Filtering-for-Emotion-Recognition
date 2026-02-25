import torch
import torch.nn as nn
from encoder import EEGEncoder, FacialEncoder

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.attn_eeg_from_face = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_face_from_eeg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_eeg = nn.LayerNorm(embed_dim)
        self.norm_face = nn.LayerNorm(embed_dim)

    def forward(self, eeg, facial):
        eeg2, _ = self.attn_eeg_from_face(eeg, facial, facial)
        facial2, _ = self.attn_face_from_eeg(facial, eeg, eeg)
        eeg = self.norm_eeg(eeg + eeg2)
        facial = self.norm_face(facial + facial2)
        return eeg, facial

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def forward(self, x):
        cls = self.cls_token.repeat(x.size(0), 1, 1)  # [B,1,D]
        x = torch.cat([cls, x], dim=1)
        out, _ = self.attn(x, x, x)
        return out[:, 0, :]  # [B,D]

class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim=128, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 二分类
        )
    def forward(self, x):
        return self.head(x)

class MultimodalFusionModel_Binary(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.eeg_encoder = EEGEncoder(embed_dim=embed_dim)
        self.facial_encoder = FacialEncoder(embed_dim=embed_dim)
        self.fusion = CrossModalFusion(embed_dim=embed_dim)
        self.pooling = AttentionPooling(embed_dim=embed_dim)
        self.cls_head = BinaryClassificationHead(input_dim=embed_dim)

    def forward(self, eeg, facial):
        eeg_feat = self.eeg_encoder(eeg)                                 # [B,T,D]
        facial_feat = self.facial_encoder(facial, tgt_len=eeg_feat.size(1))  # [B,T,D]
        eeg_feat, facial_feat = self.fusion(eeg_feat, facial_feat)
        fused = (eeg_feat + facial_feat) / 2
        pooled = self.pooling(fused)                                     # [B,D]
        return {'logits': self.cls_head(pooled)}                         # [B,2]
