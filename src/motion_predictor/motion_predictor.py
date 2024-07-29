import logging
import math

import torch
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
from einops import rearrange
from torch import nn
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder

logger = logging.getLogger(__name__)

class MotionPredictor(ModelMixin, ConfigMixin):
    def __init__(self, token_dim:int=768, hidden_dim:int=1024, num_heads:int=16, num_layers:int=8, tokens_per_frame:int=16):
        super(MotionPredictor, self).__init__()
        self.tokens_per_frame = tokens_per_frame

        # Initialize layers
        # self.input_projection = nn.Linear(token_dim, hidden_dim)  # Project token to hidden dimension
        # norm_layer = nn.LayerNorm(hidden_dim)
        self.transformer = ContinuousTransformerWrapper(
            dim_in = token_dim,
            dim_out = token_dim,
            max_seq_len = 256,
            use_abs_pos_emb = False,
            attn_layers = Encoder(
                dim = hidden_dim,
                depth = num_layers,
                heads = num_heads,
                rotary_pos_emb = True,
            )
        )
        # self.transformer = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads),
        #     num_layers=num_layers,
        #     norm=norm_layer
        # )
        # self.output_projection = nn.Linear(hidden_dim, token_dim)  # Project back to token dimension

    def create_attention_mask(self, total_frames, num_tokens):
        mask = torch.zeros((total_frames * num_tokens, total_frames * num_tokens), dtype=torch.uint8, device=self.device)
        for i in range(total_frames * num_tokens):
            current_frame_index = i // num_tokens
            first_token_index = current_frame_index * num_tokens
            last_token_index = first_token_index + num_tokens - 1
            mask[i, first_token_index] = 0
            mask[i, last_token_index] = 0
            if i == first_token_index or i == last_token_index:
                mask[i, :] = 1
        return mask.bool()

    def interpolate_tokens(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor, total_frames:int):
        # Linear interpolation in the token space
        interpolation_steps = torch.linspace(0, 1, steps=total_frames, device=start_tokens.device, dtype=torch.float16)[:, None, None]
        start_tokens_expanded = start_tokens.unsqueeze(1)  # Shape becomes [batch_size, 1, tokens, token_dim]
        end_tokens_expanded = end_tokens.unsqueeze(1)      # Shape becomes [batch_size, 1, tokens, token_dim]
        interpolated_tokens = (start_tokens_expanded * (1 - interpolation_steps) + end_tokens_expanded * interpolation_steps)
        return interpolated_tokens  # Shape: [batch_size, total_frames, tokens, token_dim]

    def forward(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor, total_frames:int):
        start_tokens = start_tokens.to(self.device)
        end_tokens = end_tokens.to(self.device)

        # Get interpolated tokens
        interpolated_tokens = self.interpolate_tokens(start_tokens, end_tokens, total_frames)

        # Flatten frames and tokens dimensions
        batch_size, total_frames, num_tokens, token_dim = interpolated_tokens.shape

        # Apply input projection
        # projected_tokens = self.input_projection(interpolated_tokens)

        # Reshape to match the transformer expected input [seq_len, batch_size, hidden_dim]
        interpolated_tokens = rearrange(interpolated_tokens, 'b f t d -> (f t) b d')

        # Create an attention mask that only allows attending to the first and last frame
        attention_mask = self.create_attention_mask(total_frames, num_tokens)

        # Transformer predicts the motion along the new sequence dimension
        logger.debug(f"interpolated_tokens {interpolated_tokens.shape}")
        motion_tokens = self.transformer(interpolated_tokens)

        # Reshape back and apply output projection
        motion_tokens = rearrange(motion_tokens, '(f t) b d -> b f t d', t=num_tokens, f=total_frames)
        # motion_tokens = self.output_projection(motion_tokens)

        motion_tokens[:, 0] = start_tokens
        motion_tokens[:, -1] = end_tokens

        return motion_tokens
