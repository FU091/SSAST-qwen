import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.vision_transformer import Block

# 完美復刻官方 PatchEmbed，解決 timm 版本相容性問題
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    def __init__(self, label_dim=12, fshape=16, tshape=16, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=1024, model_size='base'):
        super(ASTModel, self).__init__()
        
        # 根據報錯訊息 [1, 1214, 768]，你的權重檔是 base 模型 (768維)
        if model_size == 'base':
            self.embed_dim = 768
            self.depth = 12
            self.num_heads = 12
        else:
            self.embed_dim = 384
            self.depth = 12
            self.num_heads = 6

        self.v = nn.Module()
        # 初始化 PatchEmbed 並手動修正 stride
        self.v.patch_embed = PatchEmbed(img_size=(input_fdim, input_tdim), patch_size=(fshape, tshape), in_chans=1, embed_dim=self.embed_dim)
        self.v.patch_embed.proj.stride = (fstride, tstride)

        # 自動計算 patch 數量 (stride 10 會得到 12*101 = 1212)
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_out = self.v.patch_embed.proj(test_input)
        num_patches = test_out.shape[2] * test_out.shape[3]
        
        self.v.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.v.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.v.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.v.pos_drop = nn.Dropout(p=0.)

        self.v.blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=4., qkv_bias=True)
            for _ in range(self.depth)
        ])
        self.v.norm = nn.LayerNorm(self.embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, label_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 官方邏輯：輸入 [B, T, F] -> unsqueeze(1) -> [B, 1, T, F] -> transpose(2,3) -> [B, 1, F, T]
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(x.shape[0], -1, -1)
        dist_tokens = self.v.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return self.mlp_head(x)