# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import warnings
from typing import Literal

import numpy as np
import torch
# import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


approx_gelu = lambda: nn.GELU(approximate="tanh")


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def smart_pad(x: torch.Tensor, pad_len, dim=0, mode="constant", value=0, 
              pos:Literal["right", "left", "both"]="right"):
    if pad_len == 0:
        return x
    if dim < 0:
        dim += x.ndim
    assert dim < x.ndim, 'invalid padding dimension'
    pad_dim = [0, 0] * (x.ndim - dim - 1)
    if pos == "right":
        pad_dim += [0, pad_len]
    elif pos == "left":
        pad_dim += [pad_len, 0]
    else:
        pad_dim += [pad_len, pad_len]
    x = F.pad(x, pad_dim, mode=mode, value=value)
    return x


class AVSync(nn.Module):
    def __init__(
        self, hidden_size, num_heads=8, mode='merge', enable_layernorm_kernel=False
    ):
        super().__init__()
        assert mode == 'merge'
        self.mode = mode

        self.v_attn_norm = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.a_attn_norm = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.av_cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.av_mlp_norm = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.av_mlp = MLP(hidden_size)
    
    def forward(
        self, 
        audio_feat: torch.Tensor, video_feat: torch.Tensor, 
        audio_pad_mask: torch.Tensor=None, video_pad_mask: torch.Tensor=None,
    ):
        # audio_feat: [B, Ta, Sa, C], video_feat: [B, Tv, Sv, C]
        # audio_pad_mask: [B, Ta, Sa] or None, video_pad_mask: [B, Tv, Sv] or None
        device = audio_feat.device
        assert audio_feat.ndim == video_feat.ndim == 4
        assert (B := video_feat.shape[0]) == audio_feat.shape[0]
        assert (C := video_feat.shape[-1]) == audio_feat.shape[-1]
        # TODO: we assume each video and audio have comparable duration within an a-v pair
        # NOTE: Ta may be changed. Do not simply reuse it.
        Ta, Tv = audio_feat.shape[1], video_feat.shape[1]

        # split audio into frame windows with zero-paddings
        audio_feat, audio_pad_mask = self.auto_slice_audio(audio_feat, audio_pad_mask, Tv)

        # temporal align
        audio_feat = audio_feat.reshape(B*Tv, -1, C)
        video_feat = video_feat.reshape(B*Tv, -1, C)
        
        # calculate attention mask
        attn_mask = None
        if video_pad_mask is None:
            video_pad_mask = torch.full(video_feat.shape[:2], 0, dtype=torch.bool, device=device)
        else:
            video_pad_mask = video_pad_mask.reshape(B*Tv, -1)
        if audio_pad_mask is not None:
            audio_pad_mask = audio_pad_mask.reshape(B*Tv, -1)
        if audio_pad_mask.sum() > 0:
            assert (~video_pad_mask).all(), f'we assume a fixed frame number for each video but maybe slightly different durations'
            attn_mask = ~(video_pad_mask.unsqueeze(-1) | audio_pad_mask.unsqueeze(-2))

        # cross-modality attention
        v_x_m, a_x_m = self.v_attn_norm(video_feat), self.a_attn_norm(audio_feat)
        av_x_m = self.av_cross_attn(v_x_m, a_x_m)  # , attn_mask=attn_mask
        av_embds = video_feat + av_x_m

        # mlp
        av_x_m = self.av_mlp_norm(av_embds)
        av_x_m = self.av_mlp(av_x_m) 
        av_embds = av_embds + av_x_m

        av_embds = av_embds[~video_pad_mask]

        return av_embds

    def auto_slice_audio(self, audio_feat: torch.Tensor, audio_pad_mask: torch.Tensor, window_num: int):
        """
        Rearrange 1D padded tensor into a 2D tensor with zeros uniformly distributed.
        Thanks to DeepSeek-R1.
        
        Args:
            a (Tensor): Input tensor of shape (batch_size, length).
            mask (Tensor): Boolean mask tensor of shape (batch_size, length), True for valid elements.
            cols (int): Number of columns in the output 2D tensor.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, rows, cols), where rows = length // cols.
        """
        # audio_feat: [B, Ta, Sa, C], audio_pad_mask: [B, Ta, Sa]
        batch_size, Ta, Sa, C = audio_feat.shape

        av_pad_len = math.ceil(Ta / window_num) * window_num - Ta
        if av_pad_len > 0:
            if audio_pad_mask is None:
                audio_pad_mask = torch.full(audio_feat.shape[:3], 0, dtype=torch.bool, device=audio_feat.device)
            audio_feat = smart_pad(audio_feat, av_pad_len, dim=1, mode='constant', value=0.)
            audio_pad_mask = smart_pad(audio_pad_mask, av_pad_len, dim=1, mode='constant', value=True)
        Ta += av_pad_len

        valid_mask = ~audio_pad_mask[..., 0]  # [B, Ta] equal for (frequency component)
        window_size = Ta // window_num
        device = audio_feat.device
        
        # Flatten batch and sequence dimensions to handle all elements
        flat_mask = valid_mask.flatten()
        valid_elements = audio_feat.view(batch_size*Ta, Sa*C)[flat_mask]  # (total_valid, Sa*C)
        
        # Compute indices for each valid element in the original tensor
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, Ta).reshape(-1)
        batch_indices = batch_indices[flat_mask]  # (total_valid,)
        
        # Calculate the number of valid elements per sample
        n_elements = valid_mask.sum(dim=1)  # (batch_size,)
        valid_n_elements = n_elements[batch_indices]
        
        # Calculate row and column indices for valid elements
        cum_counts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), n_elements.cumsum(0)])
        local_indices = torch.arange(len(valid_elements), device=device) - cum_counts[batch_indices]
        
        # Compute row & col indices:
        rows_f = float(window_num)
        r = (local_indices.float() * rows_f / valid_n_elements.float()).floor().long()  # (total_valid,)
        k = (local_indices - r * valid_n_elements.float() / rows_f).floor().long()      # (total_valid,)
        
        # Ensure rows/columns do not exceed rows-1/cols-1
        valid_mask = (k < window_size) & (r < window_num)
        final_r = r[valid_mask]
        final_k = k[valid_mask]
        final_values = valid_elements[valid_mask]  # (total_valid, Sa*C)
        final_batch = batch_indices[valid_mask]
        
        # Create output tensor and scatter values
        output = torch.zeros((batch_size, window_num, window_size, Sa*C), device=device, dtype=audio_feat.dtype)
        output_pad_mask = torch.ones((batch_size, window_num, window_size, Sa), device=device, dtype=audio_pad_mask.dtype)
        output[final_batch, final_r, final_k] = final_values
        output_pad_mask[final_batch, final_r, final_k] = 0

        if self.mode in ['merge', 'qformer'] and (Ta - n_elements >= window_size).any():  # TODO: check
            audio_empty_window = output_pad_mask.all(dim=-1).all(dim=1)
            bi, si = torch.nonzero(audio_empty_window, as_tuple=True)
            # output[bi, :, si] = 1e-5
            output_pad_mask[bi, :, si] = 0  # TODO: avoid NaN in gradient backpropogation

        # output = output.view(batch_size, window_num * window_size, Sa, C).contiguous()
        # output_pad_mask = output_pad_mask.view(batch_size, window_num * window_size, Sa).contiguous()
        
        return output, output_pad_mask


class MLP(nn.Module):
    def __init__(self, input_size, output_size=None, hidden_size=None, act=nn.GELU()):
        super().__init__()
        if output_size is None:
            output_size = input_size
        if hidden_size is None:
            hidden_size = output_size * 4
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act,
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.proj(x)


class Modulator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = MLP(hidden_size, hidden_size*2)
    
    def forward(self, x: torch.Tensor, embed: torch.Tensor):
        shift, scale = self.proj(embed).chunk(2, dim=-1)
        return x * scale + shift


class BiMultiHeadCrossAttention(nn.Module):
    def __init__(self, m1_dim, m2_dim, embed_dim, num_heads, dropout=0.0,
                 attn_implementation: Literal['eager', 'sdpa', 'flash_attention_2']='sdpa'):
        super(BiMultiHeadCrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.m1_dim = m1_dim
        self.m2_dim = m2_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.m1_proj = nn.Linear(self.m1_dim, self.embed_dim)
        self.m2_proj = nn.Linear(self.m2_dim, self.embed_dim)
        self.values_m1_proj = nn.Linear(self.m1_dim, self.embed_dim)
        self.values_m2_proj = nn.Linear(self.m2_dim, self.embed_dim)

        self.out_m1_proj = nn.Linear(self.embed_dim, self.m1_dim)
        self.out_m2_proj = nn.Linear(self.embed_dim, self.m2_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self.rope_embeder = None

        self._reset_parameters()
        self.attn_implementation = attn_implementation

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        for proj in [
            self.m1_proj, self.values_m1_proj, self.out_m1_proj, 
            self.m2_proj, self.values_m2_proj, self.out_m2_proj
        ]:
            nn.init.xavier_uniform_(proj.weight)
            proj.bias.data.fill_(0)

    def forward(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        """_summary_

        Args:
            x1 (_type_): bs, n_m1, dim
            x2 (_type_): bs, n_m2, dim
            attention_mask_1 (_type_, optional): _description_. bs, n_m1
            attention_mask_2 (_type_, optional): _description_. bs, n_m2

        Returns:
            _type_: _description_
        """
        attn_implementation = getattr(self, 'attn_implementation', 'eager')
        if attn_implementation == 'eager':
            return self.forward_eager(x1, x2, attention_mask_1, attention_mask_2)
        elif attn_implementation == 'sdpa':
            return self.forward_sdpa(x1, x2, attention_mask_1, attention_mask_2)
        elif attn_implementation == 'flash_attention_2':
            return self.forward_flash_attn_2(x1, x2, attention_mask_1, attention_mask_2)
        else:
            raise NotImplementedError(attn_implementation)
        
    def forward_eager(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        bsz, L1, _ = x1.size()
        device = x1.device

        query_states = self.m1_proj(x1) * self.scale                    # shape(B, L1, C)
        key_states = self._shape(self.m2_proj(x2), -1, bsz)             # shape(B, h, L2, d)
        value_m1_states = self._shape(self.values_m1_proj(x1), -1, bsz) # shape(B, h, L1, d)
        value_m2_states = self._shape(self.values_m2_proj(x2), -1, bsz) # shape(B, h, L2, d)

        # shape(B*h, L, d)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, L1, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_m1_states = value_m1_states.view(*proj_shape)
        value_m2_states = value_m2_states.view(*proj_shape)

        L2 = key_states.size(1)  # L2
        # shape(B*h, L1, L2)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, L1, L2):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, L1, L2)}, "
                f"but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # shape(B*h, L2, L1)
        attn_weights_T = attn_weights.transpose(1, 2) 
        attn_weights_2 = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_2 = torch.clamp(
                attn_weights_2, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_2 = torch.clamp(
                attn_weights_2, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        if attention_mask_1 is not None or attention_mask_2 is not None:
            if attention_mask_1 is None:
                attention_mask_1 = torch.ones((bsz, L1), dtype=attention_mask_2.dtype, device=device)
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones((bsz, L2), dtype=attention_mask_1.dtype, device=device)
            # shape(L1, L2)
            mask_m2_to_m1 = attention_mask_1[:, :, None] | attention_mask_2[:, None, :]
            attn_weights.masked_fill_(torch.logical_not(mask_m2_to_m1), float("-inf"))
            # shape(L2, L1)
            mask_m1_to_m2 = mask_m2_to_m1.transpose(1, 2)
            attn_weights_2.masked_fill_(torch.logical_not(mask_m1_to_m2), float("-inf"))

        attn_probs_1 = attn_weights.softmax(dim=-1)
        attn_probs_2 = attn_weights_2.softmax(dim=-1)

        # shape(B*h, L1, L2)
        attn_probs_1 = F.dropout(attn_probs_1, p=self.dropout, training=self.training)
        # shape(B*h, L2, L1)
        attn_probs_2 = F.dropout(attn_probs_2, p=self.dropout, training=self.training)

        # shape(B*h, L1, L2) @ shape(B*h, L2, d) -> shape(B*h, L1, d)
        attn_output_1 = torch.bmm(attn_probs_1, value_m2_states)
        # shape(B*h, L2, L1) @ shape(B*h, L1, d) -> shape(B*h, L2, d)
        attn_output_2 = torch.bmm(attn_probs_2, value_m1_states)

        if attn_output_1.size() != (bsz * self.num_heads, L1, self.head_dim):
            raise ValueError(
                f"`attn_output_1` should be of size {(bsz, self.num_heads, L1, self.head_dim)}, "
                f"but is {attn_output_1.size()}"
            )

        if attn_output_2.size() != (bsz * self.num_heads, L2, self.head_dim):
            raise ValueError(
                f"`attn_output_2` should be of size {(bsz, self.num_heads, L2, self.head_dim)}, "
                f"but is {attn_output_2.size()}"
            )

        attn_output_1 = attn_output_1.view(bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2

    def forward_sdpa(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        bsz, L1, _ = x1.size()
        L2 = x2.size(1)
        device = x1.device

        query_states = self._shape(self.m1_proj(x1), -1, bsz)                # shape(B, h, L1, d)
        key_states = self._shape(self.m2_proj(x2), -1, bsz)                  # shape(B, h, L2, d)
        value_m1_states = self._shape(self.values_m1_proj(x1), -1, bsz)      # shape(B, h, L1, d)
        value_m2_states = self._shape(self.values_m2_proj(x2), -1, bsz)      # shape(B, h, L2, d)

        if attention_mask_1 is None and attention_mask_2 is None:
            mask_m1_to_m2, mask_m2_to_m1 = None, None
        else:
            if attention_mask_1 is None:
                attention_mask_1 = torch.ones((bsz, L1), dtype=attention_mask_2.dtype, device=device)
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones((bsz, L2), dtype=attention_mask_1.dtype, device=device)
            # shape(L1, L2)
            mask_m2_to_m1 = attention_mask_1[:, None, :, None] | attention_mask_2[:, None, None, :]
            # shape(L2, L1)
            mask_m1_to_m2 = mask_m2_to_m1.transpose(-1, -2)
        
        # import pdb; pdb.set_trace()
        attn_output_1 = F.scaled_dot_product_attention(
            query_states, key_states, value_m2_states,
            attn_mask=mask_m2_to_m1, dropout_p=self.dropout
        )
        attn_output_2 = F.scaled_dot_product_attention(
            key_states, query_states, value_m1_states,
            attn_mask=mask_m1_to_m2, dropout_p=self.dropout
        )

        attn_output_1 = attn_output_1.view(bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2

    def forward_flash_attn_2(self, x1, x2, attention_mask_1:torch.Tensor=None, attention_mask_2:torch.Tensor=None):
        bsz, L1, _ = x1.size()
        L2 = x2.size(1)

        if L1 <= bsz:  # copy from Attention block
            warnings.warn(f'Sequence length {L1} less than batch size {bsz}. Back to sdpa.')
            return self.forward_sdpa(x1, x2, attention_mask_1, attention_mask_2)

        if attention_mask_1 is not None and attention_mask_2 is not None:
            assert attention_mask_1.all() or attention_mask_2.all(), \
                'Currently does not support 2-directional mask attention'
        if attention_mask_1 is not None:
            x1 = x1 * attention_mask_1[..., None].type_as(x1)
        if attention_mask_2 is not None:
            x2 = x2 * attention_mask_2[..., None].type_as(x2)

        query_states = self._shape(self.m1_proj(x1), -1, bsz)                # shape(B, h, L1, d)
        key_states = self._shape(self.m2_proj(x2), -1, bsz)                  # shape(B, h, L2, d)
        value_m1_states = self._shape(self.values_m1_proj(x1), -1, bsz)      # shape(B, h, L1, d)
        value_m2_states = self._shape(self.values_m2_proj(x2), -1, bsz)      # shape(B, h, L2, d)

        from flash_attn import flash_attn_func

        # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_m1_states = value_m1_states.permute(0, 2, 1, 3)
        value_m2_states = value_m2_states.permute(0, 2, 1, 3)

        # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
        attn_output_1 = flash_attn_func(
            query_states, key_states, value_m2_states,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2)
        attn_output_2 = flash_attn_func(
            key_states, query_states, value_m1_states,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2)

        attn_output_1 = attn_output_1.view(bsz, self.num_heads, L1, self.head_dim)
        attn_output_1 = attn_output_1.transpose(1, 2)
        attn_output_1 = attn_output_1.reshape(bsz, L1, self.embed_dim)

        attn_output_2 = attn_output_2.view(bsz, self.num_heads, L2, self.head_dim)
        attn_output_2 = attn_output_2.transpose(1, 2)
        attn_output_2 = attn_output_2.reshape(bsz, L2, self.embed_dim)

        attn_output_1 = self.out_m1_proj(attn_output_1)
        attn_output_2 = self.out_m2_proj(attn_output_2)

        return attn_output_1, attn_output_2


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, attn_impl="sdpa"):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = attn_drop

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        MHCA_IMPL_DICT = {
            'sdpa': F.scaled_dot_product_attention,
            'eager': self.eager_attention,
            'flash_attention_2': self.flash_attention_2,
        }
        self.attn_func = MHCA_IMPL_DICT[attn_impl]

    def forward(self, x, cond, attn_mask=None):
        # query/value: img tokens; key: condition; mask: if valid indices
        B, N, C = x.shape
        S = cond.shape[1]

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # (B, L, num_head, head_dim) -> (B, num_head, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # (B,L) -> (B,1,L)

        x = self.attn_func(q, k, v, attn_mask, dropout_p=self.dropout_p)

        x = x.transpose(1, 2).contiguous()
        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def eager_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        attn_mask: torch.Tensor=None, dropout_p: float=0.0
    ):
        # q,k,v: shape (B, num_head, L, head_dim)
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        # shape (B, num_head, L, L)
        attn_weight = query @ key.transpose(-2, -1)
        if attn_mask is not None:
            attn_weight.masked_fill_(attn_mask.logical_not(), torch.finfo(query.dtype).min)
        attn_scores = attn_weight.softmax(-1)    # shape (B, num_head, L, L)
        attn_scores = F.dropout(attn_scores, p=dropout_p)
        z = attn_scores @ value            # shape (B, num_head, L, head_dim)
        return z
    
    @staticmethod
    def flash_attention_2(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        attn_mask: torch.Tensor=None, dropout_p: float=0.0
    ):
        B, h, L1, d = query.size()
        L2 = key.size(1)

        if L1 <= B:  # copy from Attention block
            warnings.warn(f'Sequence length {L1} less than batch size {B}. Back to sdpa.')
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
            )

        if attn_mask is not None:
            # query = query * attn_mask[..., None].type_as(query)
            assert attn_mask.shape[-1] == L2
            key = key * attn_mask[..., None].type_as(key)
            value = value * attn_mask[..., None].type_as(value)

        from flash_attn import flash_attn_func

        # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
        attn_output = flash_attn_func(
            query, key, value, dropout_p=dropout_p,
        ).transpose(1, 2)

        return attn_output
        


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    pass