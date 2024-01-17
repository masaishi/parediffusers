import torch
from torch import nn
from typing import Optional, Dict, Any
from .resnet import PareResnetBlock2D
from .transformer import PareTransformer2DModel
from .attension import PareAttention

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
	

class PareUNetMidBlock2D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_time_scale_shift: str = "default",  # default, spatial
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		attn_groups: Optional[int] = None,
		add_attention: bool = True,
		attention_head_dim: int = 1,
		output_scale_factor: float = 1.0,
	):
		super().__init__()
		resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
		self.add_attention = add_attention

		if attn_groups is None:
			attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

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

		if attention_head_dim is None:
			attention_head_dim = in_channels

		for _ in range(num_layers):
			if self.add_attention:
				attentions.append(
					PareAttention(
						in_channels,
						heads=in_channels // attention_head_dim,
						dim_head=attention_head_dim,
						rescale_output_factor=output_scale_factor,
						eps=resnet_eps,
						norm_num_groups=attn_groups,
						spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
						residual_connection=True,
						bias=True,
						upcast_softmax=True,
						_from_deprecated_attn_block=True,
					)
				)
			else:
				attentions.append(None)

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

	def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
		hidden_states = self.resnets[0](hidden_states, temb)
		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			if attn is not None:
				hidden_states = attn(hidden_states, temb=temb)
			hidden_states = resnet(hidden_states, temb)

		return hidden_states