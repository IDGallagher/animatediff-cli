import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import HunyuanAttnProcessor2_0
from diffusers.utils import BaseOutput
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn
from torch._dynamo import allow_in_graph


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError

def get_1d_rotary_pos_embed(dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0, use_real=False, trained_length: int = 16):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    if isinstance(pos, int):
        pos = np.arange(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
        rotary_position_encoding: bool = False,
        rotary_max_freq: int = 48,
    ):
        super().__init__()
        self.rotary_position_encoding = rotary_position_encoding
        self.rotary_max_freq = rotary_max_freq
        self.attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div
        self.temporal_transformer = TemporalTransformer3DModel(
        # self.temporal_transformer = TemporalTransformer3DModelModified(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            rotary_position_encoding=rotary_position_encoding,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        # print(f"Input tensor shape {input_tensor.shape}")

        rotary_embed = None
        if self.rotary_position_encoding:
            rotary_embed = RotaryEmbedding(dim = self.attention_head_dim, freqs_for='pixel', max_freq=self.rotary_max_freq).to(input_tensor.device)
            # rotary_embed = RotaryEmbedding(dim = self.attention_head_dim).to(input_tensor.device)
#  get_1d_rotary_pos_embed(self.attention_head_dim, input_tensor.shape[2], use_real=True)

        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, rotary_embed)

        output = hidden_states
        return output


@allow_in_graph
class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        rotary_position_encoding: bool = False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rotary_position_encoding=rotary_position_encoding,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        rotary_embed: Optional[RotaryEmbedding] = None,
    ):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, rotary_embed=rotary_embed,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
        )

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output

@allow_in_graph
class TemporalTransformer3DModelModified(TemporalTransformer3DModel):

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        rotary_embed: Optional[RotaryEmbedding] = None,
    ):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]

        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        batch, channel, height, weight = hidden_states.shape

        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length,  rotary_embed=rotary_embed
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
        )

        output = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length)
        output = output + residual
        return output

@allow_in_graph
class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: int = 768,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        cross_frame_attention_mode=None,
        temporal_position_encoding: bool = False,
        temporal_position_encoding_max_len: int = 24,
        rotary_position_encoding: bool = False,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rotary_position_encoding=rotary_position_encoding,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, rotary_embed: Optional[RotaryEmbedding] = None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                    rotary_embed=rotary_embed,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout: float = 0.0, max_len: int = 24):
        super().__init__()
        self.dropout: nn.Module = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe: Tensor = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@allow_in_graph
class VersatileAttention(Attention):
    def __init__(
        self,
        attention_mode: str = None,
        cross_frame_attention_mode: Optional[str] = None,
        temporal_position_encoding: bool = False,
        temporal_position_encoding_max_len: int = 24,
        rotary_position_encoding: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if attention_mode.lower() != "temporal":
            raise ValueError(f"Attention mode {attention_mode} is not supported.")
        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        self.pos_encoder = None
        if (temporal_position_encoding and attention_mode == "Temporal"):
            if rotary_position_encoding:
                # This processor is identical to the default apart from adding rotary embedding
                # self.processor = HunyuanAttnProcessor2_0()
                self.processor = RopeAttnProcessor2_0()
            else:
                self.pos_encoder = PositionalEncoding(kwargs["query_dim"], dropout=0.0, max_len=temporal_position_encoding_max_len)

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states=None, attention_mask=None, video_length=None, rotary_embed: Optional[RotaryEmbedding] = None
    ):
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )
        else:
            raise NotImplementedError

        # attention processor makes this easy so that's nice
        if rotary_embed is not None:
            hidden_states = self.processor(self, hidden_states, rotary_embed, encoder_hidden_states, attention_mask)
        else:
            hidden_states = self.processor(self, hidden_states, encoder_hidden_states, attention_mask)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states

# class VersatileAttention(Attention):
#     def __init__(
#             self,
#             attention_mode                     = None,
#             cross_frame_attention_mode         = None,
#             temporal_position_encoding         = False,
#             temporal_position_encoding_max_len = 24,
#             rotary_position_encoding: bool = False,
#             *args, **kwargs
#         ):
#         super().__init__(*args, **kwargs)
#         assert attention_mode == "Temporal"

#         self.attention_mode = attention_mode
#         self.is_cross_attention = kwargs["cross_attention_dim"] is not None

#         self.pos_encoder = PositionalEncoding(
#             kwargs["query_dim"],
#             dropout=0.,
#             max_len=temporal_position_encoding_max_len
#         ) if (temporal_position_encoding and attention_mode == "Temporal") else None

#     def extra_repr(self):
#         return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

#     def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, rotary_embed: Optional[RotaryEmbedding] = None):
#         batch_size, sequence_length, _ = hidden_states.shape

#         if self.attention_mode == "Temporal":
#             d = hidden_states.shape[1]
#             hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

#             if self.pos_encoder is not None:
#                 hidden_states = self.pos_encoder(hidden_states)

#             encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
#         else:
#             raise NotImplementedError

#         encoder_hidden_states = encoder_hidden_states

#         if self.group_norm is not None:
#             hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = self.to_q(hidden_states)
#         dim = query.shape[-1]
#         query = self.head_to_batch_dim(query)

#         if self.added_kv_proj_dim is not None:
#             raise NotImplementedError

#         encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#         key = self.to_k(encoder_hidden_states)
#         value = self.to_v(encoder_hidden_states)

#         key = self.head_to_batch_dim(key)
#         value = self.head_to_batch_dim(value)

#         if attention_mask is not None:
#             if attention_mask.shape[-1] != query.shape[1]:
#                 target_length = query.shape[1]
#                 attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#                 attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

#         # # attention, what we cannot get enough of
#         # if self._use_memory_efficient_attention_xformers:
#         #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
#         #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#         #     hidden_states = hidden_states.to(query.dtype)
#         # else:
#         #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#         #         hidden_states = self._attention(query, key, value, attention_mask)
#         #     else:
#         #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

#         attention_probs = self.get_attention_scores(query, key, attention_mask)
#         hidden_states = torch.bmm(attention_probs, value)
#         hidden_states = self.batch_to_head_dim(hidden_states)

#         # linear proj
#         hidden_states = self.to_out[0](hidden_states)

#         # dropout
#         hidden_states = self.to_out[1](hidden_states)

#         if self.attention_mode == "Temporal":
#             hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

#         return hidden_states

class RopeAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[RotaryEmbedding],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            print(deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply RoPE if needed
        if rotary_emb is not None:
            query = rotary_emb.rotate_queries_or_keys(query)
            if not attn.is_cross_attention:
                key = rotary_emb.rotate_queries_or_keys(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
