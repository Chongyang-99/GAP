"""
GAP Policy
Uses pre-extracted DinoV3 and Pi3 features for vision encoding and transformer decoder for action denoising.

Simplified architecture:
1. Load pre-extracted DinoV3 and Pi3 features (no runtime extraction)
2. State encoder processes robot joint states
3. Transformer decoder attends to all patch tokens + state token for action prediction

Updated to align with ACT-DP-TP:
- Custom decoder layer supporting separate query_pos and memory_pos
- Learnable position embedding for action queries
- Diffusion timestep added to memory (not query)

Feature extraction mode:
- Pre-extracted features from process_data.py are used directly
- No runtime feature extraction (much faster training and inference)
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
from torch import Tensor

from gap_policy.model.common.normalizer import LinearNormalizer
from gap_policy.policy.base_policy import BasePolicy
from gap_policy.common.pytorch_util import dict_apply
from gap_policy.common.model_util import print_params


# ============================================================================
# Custom Transformer Decoder supporting separate positional encodings
# (Aligned with ACT-DP-TP implementation)
# ============================================================================

class TransformerDecoderLayerWithPE(nn.Module):
    """
    Custom Transformer Decoder Layer that supports separate positional encodings
    for query and memory, following ACT-DP-TP's design.

    Position encodings are only added to Q and K in attention, not to V.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add position embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        # Position encodings (separate from features)
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: [B, N_query, D]
            memory: [B, N_memory, D]
            memory_pos: [B, N_memory, D] - Memory positional encoding
            query_pos: [B, N_query, D] - Query positional encoding
        """
        if self.norm_first:
            # Pre-norm (used in GAP)
            # Self-attention
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(
                q, k, tgt2,  # value不加位置编码
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),  # query加位置
                key=self.with_pos_embed(memory, memory_pos),  # key加位置
                value=memory,  # value不加位置！
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)

            # FFN
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-norm (ACT-DP-TP uses this)
            # Self-attention
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(
                q, k, tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # Cross-attention
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, memory_pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # FFN
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderWithPE(nn.Module):
    """Custom Transformer Decoder supporting separate positional encodings"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        # Position encodings
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: [B, N_query, D]
            memory: [B, N_memory, D]
            memory_pos: [B, N_memory, D]
            query_pos: [B, N_query, D]
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=memory_pos,
                query_pos=query_pos,
            )

        return output


# ============================================================================
# Transformer Encoder (from ACT-DP-TP) for context encoding
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer with positional encoding support.
    Adapted from ACT-DP-TP.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add position embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [B, N, D]
            pos: [B, N, D] - positional encoding
        """
        if self.norm_first:
            # Pre-norm
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(
                q, k, src2,  # value不加位置编码
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
            src = src + self.dropout1(src2)

            # FFN
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-norm
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(
                q, k, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # FFN
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder stack"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [B, N, D]
            pos: [B, N, D]
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        return output


class GAPPolicy(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # Feature dimensions (from pre-extracted features)
        dinov3_feature_dim=1024,  # DinoV3 ViT-L/16 feature dimension
        dinov3_num_views=1,       # Number of camera views for DinoV3
        # Pi3 feature config
        use_pi3_features=True,    # Whether to use pre-extracted pi3 features
        pi3_feature_dim=1024,     # Pi3 point_hidden feature dimension
        pi3_num_views=1,          # Number of camera views for Pi3 (head, left_wrist, right_wrist)
        pi3_embed_dim=1024,       # Embedding dimension for pi3 features (match with vision_feat_dim)
        # State encoder config
        state_dim=14,
        state_embed_dim=1024,     # Must match feature dimensions
        # Transformer encoder config (for context encoding - ACT-DP-TP style)
        encoder_depth=2,
        encoder_heads=8,
        encoder_dim_feedforward=2048,
        encoder_dropout=0.1,
        # Transformer decoder config (for denoising)
        decoder_depth=4,
        decoder_heads=8,
        decoder_dim_feedforward=2048,
        decoder_dropout=0.1,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # Parse action shape
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # Store feature dimensions (no vision encoder, use pre-extracted features)
        self.dinov3_feature_dim = dinov3_feature_dim
        self.dinov3_num_views = dinov3_num_views
        vision_feat_dim = dinov3_feature_dim  # 1024 for vitl16

        # State encoder (agent_pos)
        self.state_encoder = nn.Linear(state_dim, state_embed_dim)

        # Pi3 feature encoder (if using pre-extracted features)
        self.use_pi3_features = use_pi3_features
        if use_pi3_features:
            self.pi3_num_views = pi3_num_views
            self.pi3_feature_dim = pi3_feature_dim
            # Learned position embeddings for pi3 features (one per view)
            self.pi3_view_pos_embed = nn.Embedding(pi3_num_views, pi3_embed_dim)
            assert pi3_embed_dim == vision_feat_dim, \
                f"Pi3 embed dim ({pi3_embed_dim}) must match vision dim ({vision_feat_dim})"

        # Check dimensions match for token concatenation
        assert vision_feat_dim == state_embed_dim, \
            f"Vision dim ({vision_feat_dim}) must match state dim ({state_embed_dim})"

        # Feature dimension for decoder (same as vision/state dim)
        self.feature_dim = vision_feat_dim  # 1024

        # CLS token (learnable global context token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        # Learned position embeddings
        self.cls_pos_embed = nn.Embedding(1, self.feature_dim)  # CLS token position
        self.state_pos_embed = nn.Embedding(1, self.feature_dim)  # State token position

        # ========== Context Encoder (ACT-DP-TP style) ==========
        # Context encoder processes [vision_patches, state] -> enriched features
        # No CLS token - following ACT-DP-TP diffusion encoder design
        encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation='gelu',
            norm_first=True,
        )
        self.context_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
        )

        # Action query position embedding (learnable, aligned with ACT-DP-TP)
        self.action_pos_embed = nn.Embedding(horizon, self.feature_dim)

        # Action embedding (for noised actions) - multi-layer for better encoding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # Timestep embedding (for diffusion timestep, will be concat as token to memory)
        self.timestep_embed = nn.Sequential(
            nn.Linear(256, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # Timestep position embedding (learned, for timestep token - ACT-DP-TP style)
        self.timestep_pos_embed = nn.Embedding(1, self.feature_dim)

        # Custom Transformer decoder (supports separate query_pos and memory_pos)
        decoder_layer = TransformerDecoderLayerWithPE(
            d_model=self.feature_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = TransformerDecoderWithPE(
            decoder_layer,
            num_layers=decoder_depth,
        )

        # Output projection for actions
        self.action_head = nn.Linear(self.feature_dim, action_dim)

        # ========== Pi3 Feature Prediction (Point Map Prediction - PMP) ==========
        # Add pi3 feature queries for predicting future frame's pi3 features
        # if use_pi3_features:
            # Pi3 spatial query structure: 17x23 grid (matching pi3 patch grid)
        self.pi3_query_height = 17
        self.pi3_query_width = 23
        self.num_pi3_queries = self.pi3_query_height * self.pi3_query_width  # 17*23 = 391 queries

        # Pi3 feature query embedding (learned queries in 2D structure)
        # Shape: [height, width, feature_dim]
        self.pi3_query_embed = nn.Parameter(
            torch.randn(self.pi3_query_height, self.pi3_query_width, self.feature_dim)
        )

        # Pi3 query position embedding (learnable, different for each query)
        # Shape: [height, width, feature_dim]
        self.pi3_query_pos_embed = nn.Parameter(
            torch.randn(self.pi3_query_height, self.pi3_query_width, self.feature_dim)
        )

        # Pi3 feature head (output projection)
        # Each query predicts one patch's features for all views
        # Output: [height*width, num_views * pi3_feature_dim]
        self.pi3_feature_head = nn.Linear(self.feature_dim, pi3_num_views * pi3_feature_dim)

        cprint(f"  [PMP] Pi3 feature prediction enabled", "yellow")
        cprint(f"  [PMP] Pi3 query grid: {self.pi3_query_height}x{self.pi3_query_width} = {self.num_pi3_queries} queries", "yellow")
        cprint(f"  [PMP] Pi3 output per query: {pi3_num_views} views * {pi3_feature_dim} dim", "yellow")

        # Noise scheduler
        self.noise_scheduler = noise_scheduler

        # Normalizer
        self.normalizer = LinearNormalizer()

        # Store config
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        cprint("[GAP] Configuration:", "cyan")
        cprint(f"  Vision feature dim: {vision_feat_dim}", "cyan")
        cprint(f"  State embed dim: {state_embed_dim}", "cyan")
        cprint(f"  Encoder depth: {encoder_depth} (Feature encoding)", "cyan")
        cprint(f"  Decoder depth: {decoder_depth} (Action denoising)", "cyan")
        cprint(f"  Action dim: {action_dim}", "cyan")
        cprint(f"  Horizon: {horizon}", "cyan")
        cprint(f"  N obs steps: {n_obs_steps}", "cyan")
        cprint(f"  N action steps: {n_action_steps}", "cyan")
        cprint(f"  [Architecture] DinoV3 + Transformer Encoder + Diffusion Decoder (ACT-DP-TP style)", "yellow")
        cprint(f"  [Feature Encoding] Encoder processes [vision, state] -> enriched features", "yellow")
        cprint(f"  [Positional Encoding] Vision: 2D sinusoidal (separate from features)", "green")
        cprint(f"  [Positional Encoding] State: Learned (separate from features)", "green")
        cprint(f"  [Positional Encoding] Action query: Learnable embedding", "green")
        cprint(f"  [Positional Encoding] Timestep: Learned (for timestep token)", "green")
        cprint(f"  [Diffusion Timestep] Concat as token to memory (ACT-DP-TP 'cat' mode)", "green")

        print_params(self)

    def get_sinusoidal_timestep_embedding(self, timesteps, embedding_dim=256):
        """
        Generate sinusoidal timestep embeddings (for diffusion timestep)
        Args:
            timesteps: [B] tensor of timesteps
            embedding_dim: dimension of embedding
        Returns:
            embeddings: [B, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb

    def get_sinusoidal_positional_encoding(self, seq_len, embedding_dim, device):
        """
        Generate sinusoidal positional encodings (for 1D sequence position)
        Args:
            seq_len: length of sequence (horizon)
            embedding_dim: dimension of embedding (feature_dim)
            device: torch device
        Returns:
            positional_encoding: [seq_len, embedding_dim]
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float, device=device)
            * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe = torch.zeros(seq_len, embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def get_2d_sinusoidal_positional_encoding(self, height, width, embedding_dim, device, temperature=10000):
        """
        Generate 2D sinusoidal positional encodings for vision patches.
        Similar to DETR's PositionEmbeddingSine.

        Args:
            height: height of patch grid
            width: width of patch grid
            embedding_dim: dimension of embedding (must be even)
            device: torch device
            temperature: temperature for sinusoidal encoding

        Returns:
            pos: [height*width, embedding_dim] positional encodings
        """
        # Create coordinate grids
        y_embed = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).repeat(height, 1)

        # Normalize to [0, 1]
        y_embed = y_embed / height
        x_embed = x_embed / width

        # Scale to [0, 2π]
        y_embed = y_embed * 2 * torch.pi
        x_embed = x_embed * 2 * torch.pi

        # Generate sinusoidal embeddings
        # Split embedding_dim between x and y coordinates
        dim_t = torch.arange(embedding_dim // 4, dtype=torch.float32, device=device)
        dim_t = temperature ** (2 * dim_t / (embedding_dim // 2))

        pos_x = x_embed[:, :, None] / dim_t  # [H, W, D//4]
        pos_y = y_embed[:, :, None] / dim_t  # [H, W, D//4]

        # Interleave sin and cos: [H, W, D//4] -> [H, W, D//2]
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

        # Concatenate x and y encodings: [H, W, D//2] + [H, W, D//2] -> [H, W, D]
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, D]
        pos = pos.view(-1, embedding_dim)  # [H*W, D]

        return pos

    def encode_observations(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Encode observations to memory context with transformer encoder.
        Uses pre-extracted DinoV3 and Pi3 features.

        Architecture:
        1. Load pre-extracted DinoV3 and Pi3 features
        2. Project features to common dimension
        3. Concatenate: [CLS token, DinoV3 patches (all views), Pi3 patches (all views), State]
        4. Pass through encoder to get enriched memory
        5. Memory serves as context for decoder

        Args:
            obs_dict: dict with keys:
                - 'dinov3_features': [B, N_views, num_patches, D] pre-extracted DinoV3 features
                - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                - 'agent_pos': [B, 14] robot state

        Returns:
            memory: [B, N_tokens, D] - encoder output features
            memory_pos: [B, N_tokens, D] - corresponding position encodings
        """
        agent_pos = obs_dict['agent_pos']  # [B, 14]
        dinov3_features = obs_dict['dinov3_features']  # [B, N_views, num_patches, D]
        B = agent_pos.shape[0]
        device = agent_pos.device

        # 1. Process DinoV3 features
        N_dinov3_views, N_dinov3_patches, _ = dinov3_features.shape[1:]
        # Flatten all views: [B, N_views, num_patches, D] -> [B, N_views*num_patches, D]
        dinov3_flat = dinov3_features.reshape(B, N_dinov3_views * N_dinov3_patches, -1)
        # Project features
        dinov3_encoded = dinov3_flat  # [B, N_dinov3_views*num_patches, D]

        # 2. Process Pi3 features (if available)
        pi3_encoded = None
        if self.use_pi3_features and 'pi3_features' in obs_dict:
            pi3_features = obs_dict['pi3_features']  # [B, N_views, num_patches, 1024]
            N_pi3_views, N_pi3_patches, _ = pi3_features.shape[1:]
            # Flatten all views: [B, N_views, num_patches, 1024] -> [B, N_views*num_patches, 1024]
            pi3_flat = pi3_features.reshape(B, N_pi3_views * N_pi3_patches, -1)
            # Project features
            pi3_encoded = pi3_flat  # [B, N_pi3_views*num_patches, D]

        # 3. Encode state
        state_features = self.state_encoder(agent_pos)  # [B, D]
        state_features = state_features.unsqueeze(1)  # [B, 1, D]

        # 4. Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # 5. Concatenate all features: [CLS, DinoV3_patches, Pi3_patches, State]
        features_list = [cls_tokens, state_features, dinov3_encoded]
        if pi3_encoded is not None:
            features_list.append(pi3_encoded)

        encoder_input = torch.cat(features_list, dim=1)  # [B, N_total_tokens, D]

        # 6. Generate position embeddings for all tokens
        # CLS position
        cls_pos = self.cls_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        # State position
        state_pos = self.state_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        dinov3_pos = self.get_2d_sinusoidal_positional_encoding(
            height=15,
            width=20,
            embedding_dim=self.feature_dim,
            device=device
        ).unsqueeze(0).expand(B, -1, -1)  # [B, N_dinov3_patches*N_dinov3_views, D]
        encoder_pos = torch.cat([cls_pos, state_pos, dinov3_pos], dim=1)  # [B, N_tokens, D]

        if self.use_pi3_features:
            pi3_pos = self.get_2d_sinusoidal_positional_encoding(
                height=17,
                width=23,
                embedding_dim=self.feature_dim,
                device=device
            ).unsqueeze(0).expand(B, -1, -1)  # [B, N_pi3_patches*N_pi3_views, D]
            encoder_pos = torch.cat([encoder_pos, pi3_pos], dim=1)  # [B, N_tokens, D]
        
        # 7. Pass through transformer encoder
        encoder_output = self.context_encoder(encoder_input, pos=encoder_pos)  # [B, N_tokens, D]

        # 8. Use entire encoder output as memory
        memory = encoder_output  # [B, N_tokens, D]
        memory_pos = encoder_pos  # [B, N_tokens, D]

        return memory, memory_pos

    def forward_diffusion(
        self,
        noised_actions: torch.Tensor,
        timestep: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ):
        """
        Forward pass of denoising model (ACT-DP-TP 'cat' mode)

        Timestep is concatenated as an additional token to memory (not added),
        with its own learned position embedding.

        Args:
            noised_actions: [B, horizon, action_dim] noised actions
            timestep: [B] diffusion timestep
            memory: [B, N_memory, D] memory features (without position)
            memory_pos: [B, N_memory, D] memory position encodings

        Returns:
            model_output: [B, horizon, action_dim] predicted output
                (noise if prediction_type='epsilon', clean sample if 'sample')
        """
        B = noised_actions.shape[0]
        device = noised_actions.device

        # 1. Embed noised actions (no position added yet)
        action_tgt = self.action_embed(noised_actions)  # [B, horizon, D]

        # 1b. Add pi3 feature queries (if enabled)
        if self.use_pi3_features :
            # Flatten 2D pi3 queries: [height, width, D] -> [height*width, D]
            pi3_tgt_flat = self.pi3_query_embed.reshape(-1, self.feature_dim)  # [num_pi3_queries, D]
            # Expand for batch
            pi3_tgt = pi3_tgt_flat.unsqueeze(0).expand(B, -1, -1)  # [B, num_pi3_queries, D]
            # Concatenate action queries and pi3 queries
            tgt = torch.cat([action_tgt, pi3_tgt], dim=1)  # [B, horizon+num_pi3_queries, D]
        else:
            tgt = action_tgt

        # 2. Embed diffusion timestep as a token (ACT-DP-TP 'cat' mode)
        t_emb = self.get_sinusoidal_timestep_embedding(timestep)  # [B, 256]
        t_emb = self.timestep_embed(t_emb)  # [B, D]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, D]

        # Get timestep position embedding
        t_pos = self.timestep_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # Concat timestep token to memory (as an additional token)
        memory = torch.cat([memory, t_emb], dim=1)  # [B, N_memory+1, D]
        memory_pos = torch.cat([memory_pos, t_pos], dim=1)  # [B, N_memory+1, D]

        # 3. Get learnable query position embeddings
        action_query_pos = self.action_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, horizon, D]

        if self.use_pi3_features :
            # Use learned position embeddings for pi3 queries (different for each query)
            # Flatten 2D position embeddings: [height, width, D] -> [height*width, D]
            pi3_query_pos_flat = self.pi3_query_pos_embed.reshape(-1, self.feature_dim)  # [num_pi3_queries, D]
            # Expand for batch
            pi3_query_pos = pi3_query_pos_flat.unsqueeze(0).expand(B, -1, -1)  # [B, num_pi3_queries, D]
            query_pos = torch.cat([action_query_pos, pi3_query_pos], dim=1)  # [B, horizon+num_pi3_queries, D]
        else:
            query_pos = action_query_pos

        # 4. Pass through custom transformer decoder with separate positional encodings
        # Position encodings are only added in Q/K, not in V (see TransformerDecoderLayerWithPE)
        decoded = self.transformer_decoder(
            tgt=tgt,              # action + pi3 features without position
            memory=memory,        # memory with timestep token concatenated
            memory_pos=memory_pos,  # memory position (includes timestep position)
            query_pos=query_pos,    # query position (action + pi3, separate, learnable)
        )  # [B, horizon+num_pi3_queries, D]

        # 5. Split decoded features and project to respective spaces
        action_decoded = decoded[:, :self.horizon, :]  # [B, horizon, D]
        model_output = self.action_head(action_decoded)  # [B, horizon, action_dim]

        if self.use_pi3_features :
            pi3_decoded = decoded[:, self.horizon:, :]  # [B, num_pi3_queries, D]
            # Each query predicts features for all views: [B, num_pi3_queries, num_views*pi3_dim]
            pi3_output = self.pi3_feature_head(pi3_decoded)  # [B, height*width, num_views*pi3_dim]
            return model_output, pi3_output
        else:
            return model_output

    # ========= inference  ============
    def conditional_sample(
        self,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        generator=None,
        **kwargs,
    ):
        """
        Sample actions using DDPM

        Args:
            memory: [B, N_memory, D] memory features
            memory_pos: [B, N_memory, D] memory position encodings

        Returns:
            actions: [B, horizon, action_dim] denoised actions
        """
        B = memory.shape[0]
        device = memory.device
        dtype = memory.dtype

        # Start from pure noise
        actions = torch.randn(
            (B, self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Set timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising
        pi3_features_pred = None
        for t in self.noise_scheduler.timesteps:
            # Prepare timestep
            timestep = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise (and pi3 features if enabled)
            forward_output = self.forward_diffusion(
                noised_actions=actions,
                timestep=timestep,
                memory=memory,
                memory_pos=memory_pos,
            )

            if self.use_pi3_features :
                model_output, pi3_features_pred = forward_output
            else:
                model_output = forward_output

            # Denoise
            actions = self.noise_scheduler.step(
                model_output,
                t,
                actions,
            ).prev_sample

        if self.use_pi3_features :
            return actions, pi3_features_pred
        else:
            return actions

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict actions from observations (single frame)

        Args:
            obs_dict: dict with keys:
                - 'dinov3_features': [B, N_views, num_patches, D] pre-extracted DinoV3 features
                - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                - 'agent_pos': [B, 14] robot state

        Returns:
            result: dict with 'action' key
        """
        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # Encode observations to memory context (single frame)
        memory, memory_pos = self.encode_observations(nobs)

        # Sample actions (and pi3 features if enabled)
        sample_output = self.conditional_sample(
            memory=memory,
            memory_pos=memory_pos,
            **self.kwargs,
        )

        if self.use_pi3_features :
            naction_pred, pi3_pred = sample_output
        else:
            naction_pred = sample_output

        # Unnormalize
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Extract action steps to execute
        action = action_pred[:, :self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        if self.use_pi3_features :
            result["pi3_features_pred"] = pi3_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute diffusion loss (single frame per batch)

        Args:
            batch: dict with 'obs' and 'action'
                obs: dict with keys:
                    - 'dinov3_features': [B, N_views, num_patches, D] pre-extracted DinoV3 features
                    - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                    - 'agent_pos': [B, 14] robot state
                action: [B, horizon, action_dim]

        Returns:
            loss: scalar tensor
            loss_dict: dict with loss components
        """
        # Normalize
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        B = nactions.shape[0]

        # Encode observations to memory (single frame)
        memory, memory_pos = self.encode_observations(nobs)  # [B, N_memory, D]

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=nactions.device,
        ).long()

        # Sample noise
        noise = torch.randn_like(nactions)

        # Add noise to actions
        noised_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        # Predict noise (and pi3 features if enabled)
        forward_output = self.forward_diffusion(
            noised_actions=noised_actions,
            timestep=timesteps,
            memory=memory,
            memory_pos=memory_pos,
        )

        if self.use_pi3_features :
            pred, pi3_pred = forward_output
        else:
            pred = forward_output

        # Action loss
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = nactions
        elif pred_type == "v_prediction":
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * nactions
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        action_loss = F.mse_loss(pred, target, reduction="none")
        action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
        action_loss = action_loss.mean()

        loss_dict = {
            "action_loss": action_loss.item(),
        }

        # Pi3 feature prediction loss (if enabled)
        if self.use_pi3_features and "future_pi3_features" in batch:
            # Get ground truth future pi3 features (last frame of action chunk)
            future_pi3 = batch["future_pi3_features"]  # [B, N_views, num_patches, pi3_dim]

            # Reshape ground truth to match prediction shape
            # pred: [B, height*width, num_views*pi3_dim]
            # GT: [B, N_views, num_patches, pi3_dim] -> [B, num_patches, N_views*pi3_dim]
            B_pi3, N_views, num_patches, pi3_dim = future_pi3.shape

            # Transpose and reshape: [B, N_views, num_patches, pi3_dim] -> [B, num_patches, N_views, pi3_dim]
            future_pi3_transposed = future_pi3.permute(0, 2, 1, 3)  # [B, num_patches, N_views, pi3_dim]
            # Flatten last two dims: [B, num_patches, N_views*pi3_dim]
            future_pi3_flat = future_pi3_transposed.reshape(B_pi3, num_patches, N_views * pi3_dim)

            # Compute pi3 feature loss
            pi3_loss = F.mse_loss(pi3_pred, future_pi3_flat, reduction="mean")

            loss_dict["pi3_loss"] = pi3_loss.item()

            # Combined loss with weighting
            pi3_loss_weight = 0.1  # Weight for pi3 loss (auxiliary task)
            loss = action_loss + pi3_loss_weight * pi3_loss
            loss_dict["total_loss"] = loss.item()
        else:
            loss = action_loss

        return loss, loss_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
