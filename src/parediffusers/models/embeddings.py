import math
from typing import Optional
import torch
import torch.nn as nn

def get_pare_timestep_embedding(
	timesteps: torch.Tensor,
	embedding_dim: int,
	flip_sin_to_cos: bool = False,
	downscale_freq_shift: float = 1,
	scale: float = 1,
	max_period: int = 10000,
):
	half_dim = embedding_dim // 2
	exponent = -math.log(max_period) * torch.arange(
		start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
	)
	exponent = exponent / (half_dim - downscale_freq_shift)

	emb = torch.exp(exponent)
	emb = timesteps[:, None].float() * emb[None, :]

	# scale embeddings
	emb = scale * emb

	# concat sine and cosine embeddings
	emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

	# flip sine and cosine embeddings
	if flip_sin_to_cos:
		emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

	# zero pad
	if embedding_dim % 2 == 1:
		emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
	return emb


class PareTimestepEmbedding(nn.Module):
	def __init__(
		self,
		in_channels: int,
		time_embed_dim: int,
		act_fn: str = "silu",
		out_dim: int = None,
		post_act_fn: Optional[str] = None,
		cond_proj_dim=None,
		sample_proj_bias=True,
	):
		super().__init__()
		linear_cls = nn.Linear
		self.linear_1 = linear_cls(in_channels, time_embed_dim, sample_proj_bias)

		if cond_proj_dim is not None:
			self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
		else:
			self.cond_proj = None

		self.act = get_activation(act_fn)

		if out_dim is not None:
			time_embed_dim_out = out_dim
		else:
			time_embed_dim_out = time_embed_dim
		self.linear_2 = linear_cls(time_embed_dim, time_embed_dim_out, sample_proj_bias)

		if post_act_fn is None:
			self.post_act = None
		else:
			self.post_act = get_activation(post_act_fn)

	def forward(self, sample, condition=None):
		if condition is not None:
			sample = sample + self.cond_proj(condition)
		sample = self.linear_1(sample)

		if self.act is not None:
			sample = self.act(sample)

		sample = self.linear_2(sample)

		if self.post_act is not None:
			sample = self.post_act(sample)
		return sample
	
class PareTimesteps(nn.Module):
	def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
		super().__init__()
		self.num_channels = num_channels
		self.flip_sin_to_cos = flip_sin_to_cos
		self.downscale_freq_shift = downscale_freq_shift

	def forward(self, timesteps):
		t_emb = get_pare_timestep_embedding(
			timesteps,
			self.num_channels,
			flip_sin_to_cos=self.flip_sin_to_cos,
			downscale_freq_shift=self.downscale_freq_shift,
		)
		return t_emb