import math

import torch
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
from einops import rearrange
from torch import nn


def get_sinusoidal_encoding(n_positions, d_model):
    """Generate sinusoidal positional encodings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(n_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class MotionPredictor(ModelMixin, ConfigMixin):
    def __init__(self, token_dim:int=768, hidden_dim:int=1024, num_heads:int=16, num_layers:int=8, total_frames:int=16, tokens_per_frame:int=16):
        super().__init__()
        self.total_frames = total_frames
        self.tokens_per_frame = tokens_per_frame

        # Initialize layers
        self.input_projection = nn.Linear(token_dim, hidden_dim)  # Project token to hidden dimension
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(hidden_dim, token_dim)  # Project back to token dimension

        # Positional encodings
        self.token_positional_encodings = get_sinusoidal_encoding(tokens_per_frame, hidden_dim)
        self.frame_positional_encodings = get_sinusoidal_encoding(total_frames, hidden_dim)

    def interpolate_tokens(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        # Linear interpolation in the token space
        interpolation_steps = torch.linspace(0, 1, steps=self.total_frames, device=start_tokens.device, dtype=torch.float32)[:, None, None]
        start_tokens_expanded = start_tokens.unsqueeze(1)  # Shape becomes [batch_size, 1, tokens, token_dim]
        end_tokens_expanded = end_tokens.unsqueeze(1)      # Shape becomes [batch_size, 1, tokens, token_dim]
        interpolated_tokens = (start_tokens_expanded * (1 - interpolation_steps) + end_tokens_expanded * interpolation_steps)
        return interpolated_tokens  # Shape: [batch_size, total_frames, tokens, token_dim]

    def predict_motion(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        # Get interpolated tokens
        interpolated_tokens = self.interpolate_tokens(start_tokens, end_tokens)

        # Flatten frames and tokens dimensions
        batch_size, total_frames, tokens, token_dim = interpolated_tokens.shape
        seq_len = total_frames * tokens
        interpolated_tokens = interpolated_tokens.view(batch_size, seq_len, token_dim)

        # Apply input projection
        projected_tokens = self.input_projection(interpolated_tokens)

        # Add positional encodings
        # Move positional encodings to the correct device
        device = projected_tokens.device
        token_pos_enc = self.token_positional_encodings.to(device).unsqueeze(0).expand(batch_size, total_frames, -1, -1).reshape(batch_size, seq_len, -1)
        frame_pos_enc = self.frame_positional_encodings.to(device).unsqueeze(1).expand(batch_size, -1, tokens, -1).reshape(batch_size, seq_len, -1)

        # Combine both positional encodings
        combined_positional_encodings = token_pos_enc + frame_pos_enc
        projected_tokens += combined_positional_encodings

        # Reshape to match the transformer expected input [seq_len, batch_size, hidden_dim]
        projected_tokens = rearrange(projected_tokens, 'b s d -> s b d')

        # Transformer predicts the motion along the new sequence dimension
        motion_tokens = self.transformer(projected_tokens)

        # Reshape back and apply output projection
        motion_tokens = rearrange(motion_tokens, 's b d -> b s d')
        motion_tokens = self.output_projection(motion_tokens)

        # Reshape back to original frames and tokens dimensions
        motion_tokens = motion_tokens.view(batch_size, total_frames, tokens, token_dim)

        return motion_tokens

    def forward(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        return self.predict_motion(start_tokens, end_tokens)
