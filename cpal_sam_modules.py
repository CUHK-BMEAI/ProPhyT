"""
CM-CPAL SAM模块,其实就是CTP的原型检索到和image feature互动，包括1. Cross Attention融合：检索到的CTP特征与SAM image encoder输出特征融合；2. Prompt替换：检索到的CTP特征替换原始dense prompt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path
import sys


spark_path = Path(__file__).resolve().parent / "SparK" / "pretrain"
if str(spark_path) not in sys.path:
    sys.path.insert(0, str(spark_path))

try:
    from timm import create_model
    from models import resnet_1ch  # noqa: F401
    _timm_available = True
except ImportError as e:
    print(f"Warning: Failed to import timm or models: {e}")
    create_model = None
    _timm_available = False


class FPNAggregator(nn.Module):
 
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256, target_size=(64, 64)):
        super().__init__()
名
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, 3, padding=1)
        self.target_size = target_size

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: List of [B, C_i, H_i, W_i] 多层级特征

        Returns:
            aggregated: [B, out_channels, target_size[0], target_size[1]]
        """
        ups = []
        for lat, f in zip(self.lateral_convs, feats):
            x = lat(f)
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
            ups.append(x)
        x = torch.cat(ups, dim=1)
        return self.fuse_conv(x)


class PrototypeRetriever(nn.Module):

    def __init__(
        self,
        prototype_bank: np.ndarray,  # [K, D]
        latent_dim: int = 256,
        temperature: float = 0.1,
        top_k: int = 5,
        random_selection: bool = False,
        hard_weight: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.top_k = top_k
        self.random_selection = random_selection
        self.hard_weight = hard_weight
        
 
        self.register_buffer(
            'prototype_bank',
            torch.from_numpy(prototype_bank).float()
        )  # [K, D]
        
        # Query from NCCT
        self.query_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Key from CTP proto
        self.key_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, ncct_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = ncct_features.shape
        K = self.prototype_bank.shape[0]
        
    
        query = self.query_proj(ncct_features)  # [B, D]
        keys = self.key_proj(self.prototype_bank)  # [K, D]

       
        query_norm = F.normalize(query, dim=1)  # [B, D]
        keys_norm = F.normalize(keys, dim=1)  # [K, D]

        similarity = torch.matmul(query_norm, keys_norm.t())  # [B, K]
        similarity = similarity / self.temperature

        if self.random_selection: #for 消融
            indices = torch.stack([
                torch.randperm(K, device=ncct_features.device)[:self.top_k]
                for _ in range(B)
            ], dim=0)  # [B, top_k]
            sel_sim = torch.gather(similarity, 1, indices)  # [B, top_k]
            weights = F.softmax(sel_sim, dim=1)  # [B, top_k]
        else:
            # 主要的framework是用Top-K
            weights, indices = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]

            if self.hard_weight:
                #for 消融
                hard_w = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1],
                                      device=weights.device, dtype=weights.dtype)
                weights = hard_w.unsqueeze(0).expand(B, -1)  # [B, top_k]
            else:
                # 主要的framework是用soft weight
                weights = F.softmax(weights, dim=1)  # [B, top_k]

        retrieved_prototypes = self.prototype_bank[indices]  # [B, top_k, D]

        return retrieved_prototypes, weights, indices


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256, 
        latent_dim: int = 256, 
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        self.ctp_proj = nn.Linear(latent_dim, feature_dim)
        
        # Multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        sam_features: torch.Tensor,  # [B, C, H, W] SAM image encoder输出
        ctp_prototypes: torch.Tensor,  # [B, K, D] 检索到的CTP原型
        weights: torch.Tensor,  # [B, K] 检索权重
    ) -> torch.Tensor:
        B, C, H, W = sam_features.shape
        K = ctp_prototypes.shape[1]
        
      
        sam_seq = sam_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
     
        ctp_seq = self.ctp_proj(ctp_prototypes)  # [B, K, C]
        
 ）
        ctp_seq = ctp_seq * weights.unsqueeze(-1)  # [B, K, C]
        
       
        attn_out, _ = self.cross_attn(
            query=sam_seq,  # [B, H*W, C]
            key=ctp_seq,    # [B, K, C]
            value=ctp_seq   # [B, K, C]
        )  # [B, H*W, C]
        
    
        sam_seq = self.norm1(sam_seq + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(sam_seq)
        sam_seq = self.norm2(sam_seq + ffn_out)
        
       
        fused_features = sam_seq.permute(0, 2, 1).reshape(B, C, H, W)
        
        return fused_features


class PromptReplacer(nn.Module):
    """
    Prompt替换模块
    """
    def __init__(
        self,
        latent_dim: int = 256,
        prompt_embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        mode: str = 'dense',  # 'dense' or 'sparse'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.prompt_embed_dim = prompt_embed_dim
        self.image_embedding_size = image_embedding_size
        self.mode = mode

        if mode == 'dense':
            # Dense prompt: 使用两层大型Linear生成空间特征图
            H, W = image_embedding_size
            hidden_dim = 4096
            output_dim = prompt_embed_dim * H * W  # 256 * 64 * 64 = 1048576 或 256 * 16 * 16 = 65536

            self.dense_proj = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )

        elif mode == 'sparse':
            # Sparse prompt: 将CTP特征转换为点嵌入
            self.sparse_proj = nn.Sequential(
                nn.Linear(latent_dim, prompt_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(prompt_embed_dim, prompt_embed_dim)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(
        self,
        ctp_prototypes: torch.Tensor,  # [B, K, D]
        weights: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:

        B, K, D = ctp_prototypes.shape

        weighted_prototypes = torch.sum(
            ctp_prototypes * weights.unsqueeze(-1),
            dim=1
        )  # [B, D]

        if self.mode == 'dense':
            # 生成dense prompt embed
            H, W = self.image_embedding_size
            x = self.dense_proj(weighted_prototypes)  # [B, C*H*W]
            prompt_embeddings = x.reshape(B, self.prompt_embed_dim, H, W)  # [B, C, H, W]
            return prompt_embeddings

        elif self.mode == 'sparse':
            # 生成sparse prompt embed
            prompt_embeddings = self.sparse_proj(ctp_prototypes)  # [B, K, C]
            return prompt_embeddings

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class NCCTFeatureExtractor(nn.Module):

    def __init__(
        self,
        ckpt_path: str,
        feature_dim: int = 2048,
        latent_dim: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
       
        self.encoder = create_model(
            "resnet50_1ch",
            pretrained=False,
            num_classes=0,
            global_pool=""
        )
        
      
        self.fpn = self._build_fpn(latent_dim)

        
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
  
        self._load_checkpoint(ckpt_path)
        
        
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
    
    def _build_fpn(self, latent_dim: int):

        return FPNAggregator(
            in_channels=(256, 512, 1024, 2048),
            out_channels=latent_dim,
            target_size=(64, 64)
        )

    def _load_checkpoint(self, ckpt_path: str):
      
        print(f"[NCCTFeatureExtractor] 加载Stage 1b checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

    
        encoder_state = {}
        fpn_state = {}
        projector_state = {}

        for k, v in state_dict.items():
            if k.startswith('ncct_encoder.'):
                new_key = k.replace('ncct_encoder.', '')
                encoder_state[new_key] = v
            elif k.startswith('ncct_fpn.'):
                new_key = k.replace('ncct_fpn.', '')
                fpn_state[new_key] = v
            elif k.startswith('ncct_proj.'):
                new_key = k.replace('ncct_proj.', '')
                projector_state[new_key] = v

     
        if not encoder_state:
            raise ValueError(f"未在checkpoint中找到ncct_encoder权重！请检查checkpoint: {ckpt_path}")
        if not fpn_state:
            raise ValueError(f"未在checkpoint中找到ncct_fpn权重！请检查checkpoint: {ckpt_path}")
        if not projector_state:
            raise ValueError(f"未在checkpoint中找到ncct_proj权重！请检查checkpoint: {ckpt_path}")

        print(f"[NCCTFeatureExtractor] 加载encoder权重: {len(encoder_state)} 个参数")
        self.encoder.load_state_dict(encoder_state, strict=True)

        print(f"[NCCTFeatureExtractor] 加载FPN权重: {len(fpn_state)} 个参数")
        self.fpn.load_state_dict(fpn_state, strict=True)

        print(f"[NCCTFeatureExtractor] 加载projector权重: {len(projector_state)} 个参数")
        self.projector.load_state_dict(projector_state, strict=True)

        print(f"[NCCTFeatureExtractor] ✓ 所有权重加载成功（strict模式）")
    
    def _extract_multi_level_features(self, x: torch.Tensor):
       
        features = []

        # ResNet forward with intermediate outputs
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        x = self.encoder.maxpool(x)

        # Layer 1
        x = self.encoder.layer1(x)
        features.append(x)  # [B, 256, H/4, W/4]

        # Layer 2
        x = self.encoder.layer2(x)
        features.append(x)  # [B, 512, H/8, W/8]

        # Layer 3
        x = self.encoder.layer3(x)
        features.append(x)  # [B, 1024, H/16, W/16]

        # Layer 4
        x = self.encoder.layer4(x)
        features.append(x)  # [B, 2048, H/32, W/32]

        return features

    def forward(self, ncct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        multi_level_features = self._extract_multi_level_features(ncct)

       
        aggregated = self.fpn(multi_level_features)  # [B, D, 64, 64]

      
        latent = F.adaptive_avg_pool2d(aggregated, 1).flatten(1)  # [B, D]

  
        latent = self.projector(latent)  # [B, D]

       
        return multi_level_features[-1], latent


class CPALSAMWrapper(nn.Module):
    """
     整合原型检索和特征融合到SAM模型中
    """
    def __init__(
        self,
        sam_model: nn.Module,
        ncct_feature_extractor: NCCTFeatureExtractor,
        prototype_retriever: PrototypeRetriever,
        fusion_mode: str = 'cross_attention',  # 'cross_attention', 'dense_prompt', 'sparse_prompt' #主框架只是用了dense_promt
        cross_attention_fusion: Optional[CrossAttentionFusion] = None,
        prompt_replacer: Optional[PromptReplacer] = None,
    ):
        super().__init__()
        self.sam_model = sam_model
        self.ncct_feature_extractor = ncct_feature_extractor
        self.prototype_retriever = prototype_retriever
        self.fusion_mode = fusion_mode
        self.cross_attention_fusion = cross_attention_fusion
        self.prompt_replacer = prompt_replacer
        
    
        if fusion_mode == 'cross_attention' and cross_attention_fusion is None:
            raise ValueError("cross_attention模式需要提供cross_attention_fusion模块")
        if fusion_mode in ['dense_prompt', 'sparse_prompt'] and prompt_replacer is None:
            raise ValueError(f"{fusion_mode}模式需要提供prompt_replacer模块")
    
    def forward(
        self,
        images: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        B = images.shape[0]
        
     
        ncct_features, ncct_latent = self.ncct_feature_extractor(
            self._prepare_ncct_input(images)
        )
        
      
        retrieved_prototypes, weights, indices = self.prototype_retriever(ncct_latent)
        
        
        image_embeddings = self.sam_model.image_encoder(images)
        
        # 特征融合
        if self.fusion_mode == 'cross_attention':
            # Cross attention直接融合for消融
            image_embeddings = self.cross_attention_fusion(
                sam_features=image_embeddings,
                ctp_prototypes=retrieved_prototypes,
                weights=weights
            )
            
            # 使用原始prompt encoder，就是self-prompt
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=(point_coords, point_labels) if point_coords is not None else None,
                boxes=boxes,
                masks=mask_inputs
            )
        
        elif self.fusion_mode == 'dense_prompt':
            # 替换dense prompt
            sparse_embeddings, _ = self.sam_model.prompt_encoder(
                points=(point_coords, point_labels) if point_coords is not None else None,
                boxes=boxes,
                masks=None  # 不使用原始mask输入
            )
            
            # 用检索到的CTP特征生成dense prompt
            dense_embeddings = self.prompt_replacer(retrieved_prototypes, weights)
        
        elif self.fusion_mode == 'sparse_prompt':
            # 替换sparse prompt
            _, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,  # 不使用原始点
                boxes=None,   # 不使用原始框
                masks=mask_inputs
            )
            
            # 用检索到的CTP特征生成sparse prompt
            sparse_embeddings = self.prompt_replacer(retrieved_prototypes, weights)
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
      
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
       
        masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        return masks, iou_predictions, {
            'retrieved_prototypes': retrieved_prototypes,
            'weights': weights,
            'indices': indices
        }
    
    def _prepare_ncct_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            ncct = images
        else:
            ncct = images.mean(dim=1, keepdim=True)
        
        # Min-max归一化
        B = ncct.shape[0]
        ncct_flat = ncct.view(B, -1)
        min_v = ncct_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_v = ncct_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        ncct = (ncct - min_v) / (max_v - min_v + 1e-6)
        
        return ncct


def build_cpal_sam_model(
    sam_model: nn.Module,
    stage1b_ckpt_path: str,
    prototype_bank_path: str,
    fusion_mode: str = 'cross_attention',
    latent_dim: int = 256,
    top_k: int = 5,
    temperature: float = 0.1,
    freeze_ncct: bool = True,
    random_selection: bool = False,
    hard_weight: bool = False,
) -> CPALSAMWrapper:

    prototype_bank = np.load(prototype_bank_path)
    print(f"加载原型库: {prototype_bank.shape}")
    
    device = next(sam_model.parameters()).device


    ncct_feature_extractor = NCCTFeatureExtractor(
        ckpt_path=stage1b_ckpt_path,
        feature_dim=2048,
        latent_dim=latent_dim,
        freeze=freeze_ncct
    ).to(device)


    prototype_retriever = PrototypeRetriever(
        prototype_bank=prototype_bank,
        latent_dim=latent_dim,
        temperature=temperature,
        top_k=top_k,
        random_selection=random_selection,
        hard_weight=hard_weight,
    ).to(device)
    

    cross_attention_fusion = None
    prompt_replacer = None
    
    if fusion_mode == 'cross_attention':

        sam_feature_dim = sam_model.image_encoder.neck[0].out_channels if hasattr(sam_model.image_encoder, 'neck') else 256

        cross_attention_fusion = CrossAttentionFusion(
            feature_dim=sam_feature_dim,
            latent_dim=latent_dim,
            num_heads=8,
            dropout=0.1
        ).to(device)

    elif fusion_mode in ['dense_prompt', 'sparse_prompt']:
        # SAM prompt encoder,就是our框架只替换‘dense_prompt’了
        prompt_embed_dim = sam_model.prompt_encoder.embed_dim
        image_embedding_size = sam_model.prompt_encoder.image_embedding_size

        prompt_replacer = PromptReplacer(
            latent_dim=latent_dim,
            prompt_embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size,
            mode='dense' if fusion_mode == 'dense_prompt' else 'sparse'
        ).to(device)
    

    cpal_sam_model = CPALSAMWrapper(
        sam_model=sam_model,
        ncct_feature_extractor=ncct_feature_extractor,
        prototype_retriever=prototype_retriever,
        fusion_mode=fusion_mode,
        cross_attention_fusion=cross_attention_fusion,
        prompt_replacer=prompt_replacer
    )
    
    return cpal_sam_model
