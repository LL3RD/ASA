from typing import Tuple, Union, Sequence

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
import torch
import torch.nn as nn
from monai.networks.blocks.mlp import MLPBlock
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import numpy as np
from monai.utils import optional_import
import torch.nn.functional as F

from monai.networks.layers import Conv

einops, _ = optional_import("einops")
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option
import math

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PatchEmbeddingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int,
            num_heads: int,
            pos_embed: str,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            use_learnable_pos_emb: bool = False,
            symmetry: int = 1,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super(PatchEmbeddingBlock, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = in_channels * np.prod(patch_size)

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i + 1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len),
                nn.Linear(self.patch_dim, hidden_size),
            )
        # learnable
        self.use_learnable_posemb = use_learnable_pos_emb
        if use_learnable_pos_emb:
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
            self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        else:
            self.sym = symmetry
            if symmetry == 3:
                position_embeddings = build_3d_sincos_position_embedding(
                    [im_d // p_d for im_d, p_d in zip(img_size, patch_size)], hidden_size)
            elif symmetry == 1:
                position_embeddings = build_sincos_position_embedding(size=int(self.n_patches / 2),
                                                                           embed_dim=hidden_size,
                                                                           symm=int(img_size[2] / 2))
            else:
                position_embeddings = get_sinusoid_encoding_table(self.n_patches, hidden_size)
            self.register_buffer("position_embeddings", position_embeddings)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        x = self.patch_embeddings(x).type_as(x)
        if self.sym == 0 and self.use_learnable_posemb==False:
            embeddings = x + self.position_embeddings.expand(x.size(0), -1, -1).type_as(x).to(x.device).clone().detach()
        else:
            embeddings = x + self.position_embeddings.type_as(x).to(x.device)
        embeddings = self.dropout(embeddings)
        return embeddings


def recover_symm(pos_embed, symm):
    pos_embed_left = pos_embed[0, :, :]
    pos_embed_right = pos_embed[1, :, :]
    pos_embed_right = einops.rearrange(pos_embed_right, "(h p1) d->h p1 d", p1=symm)
    pos_embed_right = torch.flip(pos_embed_right, dims=[1])
    pos_embed_left = einops.rearrange(pos_embed_left, "(h p1) d->h p1 d", p1=symm)
    pos_embed = torch.hstack([pos_embed_left, pos_embed_right])
    pos_embed = einops.rearrange(pos_embed, "h w d->(h w) d")
    return pos_embed


def build_sincos_position_embedding(size, embed_dim, symm, temperature=100000.):
    D = size

    def get_position_angle_vec(position):
        return [position / np.power(temperature, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(D)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    pos_embed = torch.FloatTensor(sinusoid_table)
    pos_embed = torch.stack([pos_embed, pos_embed], dim=0)
    pos_embed = recover_symm(pos_embed, symm)
    pos_embed.requires_grad = False
    return pos_embed


def build_3d_sincos_position_embedding(grid_size, embed_dim, temperature=100000.):
    h, w, s = grid_size
    grid_w = torch.cat(
        [torch.arange(w / 2, dtype=torch.float32), torch.arange(w / 2, dtype=torch.float32).flip(dims=[0])], dim=0)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_s = torch.arange(s, dtype=torch.float32)
    grid_w, grid_h, grid_s = torch.meshgrid(grid_w, grid_h, grid_s)
    assert embed_dim % 6 == 0
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_s = torch.einsum('m,d->md', [grid_s.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_s), torch.cos(out_s), torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
        dim=1)[None, :, :]
    pos_emb.requires_grad = False
    # pos_emb = einops.rearrange(pos_emb, "(h w s) p-> h w s p", w=w, s=s)
    return pos_emb


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class SWABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            window_size: int = 32,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super(SWABlock, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = float(self.head_dim ** -0.5)
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))
        coords_d = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_d]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        q, k, v = einops.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads).type_as(x)
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1).type_as(x)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # W,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        att_mat = att_mat + relative_position_bias.unsqueeze(0).type_as(att_mat)

        if mask is not None:
            nW = mask.shape[0]
            att_mat = att_mat.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            att_mat = att_mat.view(-1, self.num_heads, N, N)
            att_mat = self.softmax(att_mat)
        else:
            att_mat = self.softmax(att_mat)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


def window_partition(x, window_size):
    B, D, C = x.shape
    x = x.view(B, D // window_size, window_size, C)
    windows = x.view(-1, window_size, C).contiguous()
    return windows


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            num_heads: int,
            input_resolution: int,
            dropout_rate: float = 0.0,
            window_size: int = 32,
            shift_size: int = 0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SWABlock(hidden_size, num_heads, dropout_rate, window_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.pad = (self.window_size - input_resolution % self.window_size) % self.window_size
        if self.shift_size > 0:
            img_mask = torch.zeros((1, self.input_resolution, 1))
            img_mask = F.pad(img_mask, (0, 0, 0, self.pad, 0, 0))
            d_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for d in d_slices:
                img_mask[:, d, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, self.pad, 0, 0))
        _, Lp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1,))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = attn_windows.view(B, Lp, C)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1,))
        else:
            x = shifted_x

        if self.pad > 0:
            x = x[:, :L, :].contiguous()

        x = shortcut + x.view(B, L, C)

        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            input_resolution: int,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            window_size: int = 32,
            symmetry: int = 1,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        """

        super(ViT, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            symmetry=symmetry
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, input_resolution=input_resolution,
                              dropout_rate=dropout_rate,
                              window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2) for i in
             range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())

    def forward(self, x, mask):
        x = self.patch_embedding(x)
        B, _, C = x.shape  # 2, 32768, 120
        x_vis = x[~mask].reshape(B, -1, C)
        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x_vis = blk(x_vis)
            hidden_states_out.append(x_vis)
        x_vis = self.norm(x_vis)
        if self.classification:
            x = self.classification_head(x[:, 0])
        return x_vis, hidden_states_out


class PreTrainMEDIUMEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            img_size: Tuple[int, int, int],
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "perceptron",
            dropout_rate: float = 0.0,
            mask_ratio: float = 0.75,
            symmetry: int = 1,
            window_size: int = 32,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (8, 8, 8)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        l, w, s = self.feat_size
        input_resolution = l * w * s - int(l * w * s * mask_ratio)
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            input_resolution=input_resolution,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            symmetry=symmetry,
            window_size=window_size,
        )

    def forward(self, x_in, mask):
        x, hidden_states_out = self.vit(x_in, mask)
        return x


class PretrainMEDIUMDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            img_size: Tuple[int, int, int],
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            dropout_rate: float = 0.0,
            mask_ratio: float = 0.0,
            window_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_layers = 8
        self.patch_size = (8, 8, 8)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        l, w, s = self.feat_size
        input_resolution = l * w * s - int(l * w * s * mask_ratio)
        self.hidden_size = hidden_size

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, input_resolution=input_resolution,
                              dropout_rate=dropout_rate, window_size=window_size) for i in range(self.num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(self.hidden_size,
                              in_channels * self.patch_size[0] * self.patch_size[0] * self.patch_size[0])

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]

        return x


class PretrainMEDIUMVIT(nn.Module):
    def __init__(self,
                 img_size=(128, 128, 128),
                 encoder_in_chans=1,
                 encoder_embed_dim=384,  # hidden_size
                 decoder_embed_dim=384,
                 mask_ratio=0.75,
                 use_learnable_pos_emb=False,
                 symmetry=1,
                 window_size: int = 32,
                 ):
        super().__init__()
        self.encoder = PreTrainMEDIUMEncoder(in_channels=encoder_in_chans, img_size=img_size,
                                            hidden_size=encoder_embed_dim, mlp_dim=encoder_embed_dim * 4,
                                            mask_ratio=mask_ratio, symmetry=symmetry, window_size=window_size)
        self.decoder = PretrainMEDIUMDecoder(in_channels=encoder_in_chans, img_size=img_size,
                                            hidden_size=decoder_embed_dim, mlp_dim=decoder_embed_dim * 4, mask_ratio=0, window_size=window_size)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = self.encoder.feat_size[0] * self.encoder.feat_size[1] * self.encoder.feat_size[2]
        if use_learnable_pos_emb:
            self.position_embeddings_dec = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        else:
            if symmetry == 3:
                self.position_embeddings_dec = build_3d_sincos_position_embedding(self.decoder.feat_size,
                                                                                  decoder_embed_dim)
            elif symmetry == 1:
                self.position_embeddings_dec = build_sincos_position_embedding(int(num_patches / 2), decoder_embed_dim,
                                                                               int(self.decoder.feat_size[2] / 2))
            elif symmetry == 0:
                self.position_embeddings_dec = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):

        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, N, C = x_vis.shape

        expand_pos_embed = self.position_embeddings_dec.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        x = self.decoder(x_full, pos_emd_mask.shape[1])
        return x


if __name__ == '__main__':
    device = "cuda:4"
    model = PretrainMEDIUMVIT(mask_ratio=0.40).to(device)
    x = torch.rand((2, 1, 128, 128, 128)).to(device)
    import numpy as np

    l, w, s = (int(128 / 8), int(128 / 8), int(128 / 8))
    mask = np.zeros((2, l * w * s))
    for j in range(2):
        mask[j] = np.hstack([
            np.zeros(l * w * s - int(l * w * s * 0.40)),
            np.ones(int(l * w * s * 0.40))
        ])
        np.random.shuffle(mask[j])

    mask = torch.from_numpy(mask).bool().to(device)
    y = model(x, mask)
    print(y.shape)

    import time

    time.sleep(60)
