import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .unet_2d_get_blocks import pare_get_up_block, pare_get_down_block
from .unet_2d_mid_blocks import PareUNetMidBlock2D

class PareEncoder(nn.Module):
	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		down_block_types: Tuple[str, ...] = ("PareDownEncoderBlock2D",),
		block_out_channels: Tuple[int, ...] = (64,),
		layers_per_block: int = 2,
		norm_num_groups: int = 32,
		act_fn: str = "silu",
		double_z: bool = True,
		mid_block_add_attention=True,
	):
		super().__init__()
		self.layers_per_block = layers_per_block

		self.conv_in = nn.Conv2d(
			in_channels,
			block_out_channels[0],
			kernel_size=3,
			stride=1,
			padding=1,
		)

		self.mid_block = None
		self.down_blocks = nn.ModuleList([])

		# down
		output_channel = block_out_channels[0]
		for i, down_block_type in enumerate(down_block_types):
			input_channel = output_channel
			output_channel = block_out_channels[i]
			is_final_block = i == len(block_out_channels) - 1

			down_block = pare_get_down_block(
				down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=None,
			)
			self.down_blocks.append(down_block)

		# mid
		self.mid_block = PareUNetMidBlock2D(
			in_channels=block_out_channels[-1],
			resnet_eps=1e-6,
			resnet_act_fn=act_fn,
			output_scale_factor=1,
			attention_head_dim=block_out_channels[-1],
			resnet_groups=norm_num_groups,
			temb_channels=None,
			add_attention=mid_block_add_attention,
		)

		# out
		self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
		self.conv_act = nn.SiLU()

		conv_out_channels = 2 * out_channels if double_z else out_channels
		self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

		self.gradient_checkpointing = False

	def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
		sample = self.conv_in(sample)
		
		# down
		for down_block in self.down_blocks:
			sample = down_block(sample)

		# middle
		sample = self.mid_block(sample)

		# post-process
		sample = self.conv_norm_out(sample)
		sample = self.conv_act(sample)
		sample = self.conv_out(sample)

		return sample


class PareDecoder(nn.Module):
	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		up_block_types: Tuple[str, ...] = ("PareUpDecoderBlock2D",),
		block_out_channels: Tuple[int, ...] = (64,),
		layers_per_block: int = 2,
		norm_num_groups: int = 32,
		act_fn: str = "silu",
		norm_type: str = "group",  # group, spatial
		mid_block_add_attention=True,
	):
		super().__init__()
		self.layers_per_block = layers_per_block

		self.conv_in = nn.Conv2d(
			in_channels,
			block_out_channels[-1],
			kernel_size=3,
			stride=1,
			padding=1,
		)

		self.mid_block = None
		self.up_blocks = nn.ModuleList([])

		temb_channels = in_channels if norm_type == "spatial" else None

		# mid
		self.mid_block = PareUNetMidBlock2D(
			in_channels=block_out_channels[-1],
			resnet_eps=1e-6,
			resnet_act_fn=act_fn,
			output_scale_factor=1,
			attention_head_dim=block_out_channels[-1],
			resnet_groups=norm_num_groups,
			temb_channels=temb_channels,
			add_attention=mid_block_add_attention,
		)

		# up
		reversed_block_out_channels = list(reversed(block_out_channels))
		output_channel = reversed_block_out_channels[0]
		for i, up_block_type in enumerate(up_block_types):
			prev_output_channel = output_channel
			output_channel = reversed_block_out_channels[i]

			is_final_block = i == len(block_out_channels) - 1

			up_block = pare_get_up_block(
				up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
				attention_head_dim=output_channel,
                temb_channels=temb_channels,
			)
			self.up_blocks.append(up_block)
			prev_output_channel = output_channel

		# out
		self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
		self.conv_act = nn.SiLU()
		self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

		self.gradient_checkpointing = False

	def forward(
		self,
		sample: torch.FloatTensor,
		latent_embeds: Optional[torch.FloatTensor] = None,
	) -> torch.FloatTensor:
		r"""The forward method of the `Decoder` class."""

		sample = self.conv_in(sample)

		# middle
		sample = self.mid_block(sample, latent_embeds)

		# up
		for up_block in self.up_blocks:
			sample = up_block(sample, latent_embeds)

		# post-process
		if latent_embeds is None:
			sample = self.conv_norm_out(sample)
		else:
			sample = self.conv_norm_out(sample, latent_embeds)
		sample = self.conv_act(sample)
		sample = self.conv_out(sample)

		return sample
	

class PareDiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def kl(self, other: "PareDiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean