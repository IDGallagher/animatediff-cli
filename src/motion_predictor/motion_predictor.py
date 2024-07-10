import torch
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
from torch import nn


class MotionPredictor(ModelMixin, ConfigMixin):
    def __init__(self, token_dim:int=768, hidden_dim:int=1024, num_heads:int=16, num_layers:int=8, total_frames:int=16):
        super().__init__()
        self.total_frames = total_frames
        self.input_projection = nn.Linear(token_dim, hidden_dim)  # Project token to hidden dimension
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(hidden_dim, token_dim)  # Project back to token dimension

    def interpolate_tokens(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        # Linear interpolation in the token space
        interpolation_steps = torch.linspace(0, 1, steps=self.total_frames, device=start_tokens.device, dtype=torch.float16)[:, None, None]
        interpolated_tokens = start_tokens[None, :] * (1 - interpolation_steps) + end_tokens[None, :] * interpolation_steps
        return interpolated_tokens.squeeze(0)

    def predict_motion(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        # Get interpolated tokens
        interpolated_tokens = self.interpolate_tokens(start_tokens, end_tokens)

        # Apply input projection
        interpolated_tokens = self.input_projection(interpolated_tokens)

        # Transformer predicts the motion along frame dimension
        motion_tokens = self.transformer(interpolated_tokens)
        motion_tokens = self.output_projection(motion_tokens)  # Final dimension projection
        return motion_tokens

    def forward(self, start_tokens:torch.Tensor, end_tokens:torch.Tensor):
        return self.predict_motion(start_tokens, end_tokens)
