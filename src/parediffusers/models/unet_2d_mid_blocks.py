import torch
from torch import nn
from typing import Optional, Dict, Any
from .resnet import PareResnetBlock2D
from .transformer import PareTransformer2DModel

class PareUNetMidBlock2DCrossAttn(nn.Module):
	def __init__(
		self,
		in_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		transformer_layers_per_block: int = 1,
		resnet_eps: float = 1e-6,
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		num_attention_heads: int = 1,
		output_scale_factor: float = 1.0,
		cross_attention_dim: int = 1280,
		use_linear_projection: bool = False,
		upcast_attention: bool = False,
	):
		super().__init__()

		self.has_cross_attention = True
		self.num_attention_heads = num_attention_heads
		resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

		# support for variable transformer layers per block
		if isinstance(transformer_layers_per_block, int):
			transformer_layers_per_block = [transformer_layers_per_block] * num_layers

		# there is always at least one resnet
		resnets = [
			PareResnetBlock2D(
				in_channels=in_channels,
				out_channels=in_channels,
				temb_channels=temb_channels,
				eps=resnet_eps,
				groups=resnet_groups,
				dropout=dropout,
				non_linearity=resnet_act_fn,
				output_scale_factor=output_scale_factor,
			)
		]
		attentions = []

		for i in range(num_layers):
			attentions.append(
				PareTransformer2DModel(
					num_attention_heads,
					in_channels // num_attention_heads,
					in_channels=in_channels,
					num_layers=transformer_layers_per_block[i],
					cross_attention_dim=cross_attention_dim,
					norm_num_groups=resnet_groups,
					use_linear_projection=use_linear_projection,
					upcast_attention=upcast_attention,
				)
			)
			resnets.append(
				PareResnetBlock2D(
					in_channels=in_channels,
					out_channels=in_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
				)
			)

		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)

		self.gradient_checkpointing = False

	def forward(
		self,
		hidden_states: torch.FloatTensor,
		temb: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
	) -> torch.FloatTensor:
		hidden_states = self.resnets[0](hidden_states, temb)
		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			hidden_states = attn(
				hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				cross_attention_kwargs=cross_attention_kwargs,
				attention_mask=attention_mask,
				encoder_attention_mask=encoder_attention_mask,
				return_dict=False,
			)[0]
			hidden_states = resnet(hidden_states, temb)

		return hidden_states