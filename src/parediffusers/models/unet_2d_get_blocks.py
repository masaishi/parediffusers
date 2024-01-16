from torch import nn
from typing import Optional
from .unet_2d_blocks import (
	PareDownBlock2D,
	PareCrossAttnDownBlock2D,
	PareUpBlock2D,
	PareCrossAttnUpBlock2D,
)

def pare_get_down_block(
	down_block_type: str,
	num_layers: int,
	in_channels: int,
	out_channels: int,
	temb_channels: int,
	add_downsample: bool,
	resnet_eps: float,
	resnet_act_fn: str,
	transformer_layers_per_block: int = 1,
	num_attention_heads: Optional[int] = None,
	resnet_groups: Optional[int] = None,
	cross_attention_dim: Optional[int] = None,
	downsample_padding: Optional[int] = None,
	dual_cross_attention: bool = False,
	use_linear_projection: bool = False,
	only_cross_attention: bool = False,
	upcast_attention: bool = False,
	dropout: float = 0.0,
):
	down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
	if down_block_type == "DownBlock2D":
		return PareDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
		)
	elif down_block_type == "CrossAttnDownBlock2D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
		return PareCrossAttnDownBlock2D(
			num_layers=num_layers,
			transformer_layers_per_block=transformer_layers_per_block,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			cross_attention_dim=cross_attention_dim,
			num_attention_heads=num_attention_heads,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
		)


def pare_get_up_block(
	up_block_type: str,
	num_layers: int,
	in_channels: int,
	out_channels: int,
	prev_output_channel: int,
	temb_channels: int,
	add_upsample: bool,
	resnet_eps: float,
	resnet_act_fn: str,
	resolution_idx: Optional[int] = None,
	transformer_layers_per_block: int = 1,
	num_attention_heads: Optional[int] = None,
	resnet_groups: Optional[int] = None,
	cross_attention_dim: Optional[int] = None,
	dual_cross_attention: bool = False,
	use_linear_projection: bool = False,
	only_cross_attention: bool = False,
	upcast_attention: bool = False,
	dropout: float = 0.0,
) -> nn.Module:
	up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
	if up_block_type == "UpBlock2D":
		return PareUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
		)
	elif up_block_type == "CrossAttnUpBlock2D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
		return PareCrossAttnUpBlock2D(
			num_layers=num_layers,
			transformer_layers_per_block=transformer_layers_per_block,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_groups=resnet_groups,
			cross_attention_dim=cross_attention_dim,
			num_attention_heads=num_attention_heads,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
		)
