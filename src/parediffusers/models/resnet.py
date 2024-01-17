import torch
from torch import nn
from typing import Optional
from ..utils import get_activation

class PareResnetBlock2D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: Optional[int] = None,
		temb_channels: int = 512,
		eps: float = 1e-6,
		groups: int = 32,
		groups_out: Optional[int] = None,
		dropout: float = 0.0,
		non_linearity: str = "swish",
		output_scale_factor: float = 1.0,
		skip_time_act: bool = False,
	):
		super().__init__()
		self.in_channels = in_channels
		out_channels = in_channels if out_channels is None else out_channels
		self.out_channels = out_channels
		self.output_scale_factor = output_scale_factor

		linear_cls = nn.Linear
		conv_cls = nn.Conv2d

		if groups_out is None:
			groups_out = groups

		self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

		self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

		if temb_channels is not None:
			self.time_emb_proj = linear_cls(temb_channels, out_channels)
		else:
			self.time_emb_proj = None

		self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

		self.dropout = torch.nn.Dropout(dropout)
		self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

		self.nonlinearity = get_activation(non_linearity)

		self.upsample = self.downsample = None
		self.use_in_shortcut = self.in_channels != out_channels

		self.conv_shortcut = None
		if self.use_in_shortcut:
			self.conv_shortcut = conv_cls(
				in_channels,
				out_channels,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=True,
			)

	def forward(
		self,
		input_tensor: torch.FloatTensor,
		temb: torch.FloatTensor,
	) -> torch.FloatTensor:
		hidden_states = input_tensor

		hidden_states = self.norm1(hidden_states)
		hidden_states = self.nonlinearity(hidden_states)

		if self.upsample is not None:
			if hidden_states.shape[0] >= 64:
				input_tensor = input_tensor.contiguous()
				hidden_states = hidden_states.contiguous()
			input_tensor = (
				self.upsample(input_tensor)
			)
			hidden_states = (
				self.upsample(hidden_states)
			)
		elif self.downsample is not None:
			input_tensor = (
				self.downsample(input_tensor)
			)
			hidden_states = (
				self.downsample(hidden_states)
			)

		hidden_states = self.conv1(hidden_states)

		if self.time_emb_proj is not None:
			temb = self.nonlinearity(temb)
			temb = (
				self.time_emb_proj(temb)[:, :, None, None]
			)

		if temb is not None:
			hidden_states = hidden_states + temb
		hidden_states = self.norm2(hidden_states)

		hidden_states = self.nonlinearity(hidden_states)

		hidden_states = self.dropout(hidden_states)
		hidden_states = self.conv2(hidden_states)

		if self.conv_shortcut is not None:
			input_tensor = (
				self.conv_shortcut(input_tensor)
			)

		output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

		return output_tensor