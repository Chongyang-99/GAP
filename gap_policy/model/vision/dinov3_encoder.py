"""
DinoV3 Vision Encoder
Wraps pretrained DinoV3 model for feature extraction from RGB images.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange
import os


class DINOV3(nn.Module):
    """
    DINOV3 Vision Encoder
    Extracts patch tokens from RGB images using pretrained DinoV3
    No projection - uses raw DinoV3 features
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",  # dinov3_vitl16, dinov3_vitg16, etc.
        freeze: bool = True,
        repo_dir: str = "thirdparty/dinov3",  # Local repo directory
        weights_path: str = "pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",  # Local weights
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze

        # Load pretrained DinoV3 model from local directory
        print(f"Loading {model_name} from local directory...")
        print(f"  Repo dir: {repo_dir}")
        print(f"  Weights: {weights_path}")

        self.dinov3 = torch.hub.load(
            repo_dir,
            model_name,
            source='local',
            weights=weights_path
        )

        # Get model specs
        self.patch_size = self.dinov3.patch_embed.patch_size[0]  # 16 for vitl16/vitg16
        self.embed_dim = self.dinov3.embed_dim  # 1024 for vitl, 1536 for vitg
        self.output_dim = self.embed_dim  # No projection

        # Freeze backbone if specified
        if self.freeze:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            self.dinov3.eval()
            print(f"  DinoV3 backbone frozen")
        else:
            print(f"  DinoV3 backbone trainable")

        print(f"  Model: {model_name}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embed dim: {self.embed_dim}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DinoV3 features from images

        Args:
            images: [B, 3, H, W] RGB images already normalized with ImageNet stats
                    (normalization done in dataset normalizer)

        Returns:
            features: [B, N_patches, embed_dim] patch features (without CLS token)
        """
        B, C, H, W = images.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        if self.freeze:
            self.dinov3.eval()

        # Extract features
        with torch.set_grad_enabled(not self.freeze):
            features = self.dinov3.forward_features(images) 
            return features['x_norm_patchtokens']

    def get_num_patches(self, image_size: Tuple[int, int]) -> int:
        """
        Calculate number of patches for given image size
        Args:
            image_size: (H, W)
        Returns:
            num_patches: number of patches
        """
        H, W = image_size
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        return num_patches
