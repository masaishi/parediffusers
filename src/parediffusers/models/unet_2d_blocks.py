import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from .resnet import PareResnetBlock2D
from .transformer import PareTransformer2DModel

class PareDownBlock2D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		output_scale_factor: float = 1.0,
		add_downsample: bool = True,
		downsample_padding: int = 1,
	):
		super().__init__()
		resnets = []

		for i in range(num_layers):
			in_channels = in_channels if i == 0 else out_channels
			resnets.append(
				PareResnetBlock2D(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
				)
			)

		self.resnets = nn.ModuleList(resnets)

		if add_downsample:
			self.downsamplers = nn.ModuleList(
				[
					PareDownsample2D(
						out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
					)
				]
			)
		else:
			self.downsamplers = None

		self.gradient_checkpointing = False

	def forward(
		self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
	) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
		output_states = ()

		for resnet in self.resnets:
			hidden_states = resnet(hidden_states, temb)

			output_states = output_states + (hidden_states,)

		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)

			output_states = output_states + (hidden_states,)

		return hidden_states, output_states


class PareCrossAttnDownBlock2D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		transformer_layers_per_block: int = 1,
		resnet_eps: float = 1e-6,
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		num_attention_heads: int = 1,
		cross_attention_dim: int = 1280,
		output_scale_factor: float = 1.0,
		downsample_padding: int = 1,
		add_downsample: bool = True,
		dual_cross_attention: bool = False,
		use_linear_projection: bool = False,
		only_cross_attention: bool = False,
		upcast_attention: bool = False,
	):
		super().__init__()
		resnets = []
		attentions = []

		self.has_cross_attention = True
		self.num_attention_heads = num_attention_heads
		if isinstance(transformer_layers_per_block, int):
			transformer_layers_per_block = [transformer_layers_per_block] * num_layers

		for i in range(num_layers):
			in_channels = in_channels if i == 0 else out_channels
			resnets.append(
				PareResnetBlock2D(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
				)
			)
			if not dual_cross_attention:
				attentions.append(
					PareTransformer2DModel(
						num_attention_heads,
						out_channels // num_attention_heads,
						in_channels=out_channels,
						num_layers=transformer_layers_per_block[i],
						cross_attention_dim=cross_attention_dim,
						norm_num_groups=resnet_groups,
						use_linear_projection=use_linear_projection,
						only_cross_attention=only_cross_attention,
						upcast_attention=upcast_attention,
					)
				)

		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)

		if add_downsample:
			self.downsamplers = nn.ModuleList(
				[
					PareDownsample2D(
						out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
					)
				]
			)
		else:
			self.downsamplers = None

		self.gradient_checkpointing = False

	def forward(
		self,
		hidden_states: torch.FloatTensor,
		temb: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		additional_residuals: Optional[torch.FloatTensor] = None,
	) -> [torch.FloatTensor]:
		output_states = ()

		blocks = list(zip(self.resnets, self.attentions))

		for i, (resnet, attn) in enumerate(blocks):
			hidden_states = resnet(hidden_states, temb)
			hidden_states = attn(
				hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				cross_attention_kwargs=cross_attention_kwargs,
				attention_mask=attention_mask,
				encoder_attention_mask=encoder_attention_mask,
				return_dict=False,
			)[0]

			# apply additional residuals to the output of the last pair of resnet and attention blocks
			if i == len(blocks) - 1 and additional_residuals is not None:
				hidden_states = hidden_states + additional_residuals

			output_states = output_states + (hidden_states,)

		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)

			output_states = output_states + (hidden_states,)

		return hidden_states, output_states


class PareDownsample2D(nn.Module):
	def __init__(
		self,
		channels: int,
		use_conv: bool = False,
		out_channels: Optional[int] = None,
		padding: int = 1,
		name: str = "conv",
		kernel_size=3,
		bias=True,
	):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.padding = padding
		stride = 2
		self.name = name
		conv_cls = nn.Conv2d

		self.norm = None
	
		if use_conv:
			conv = conv_cls(
				self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
			)
		else:
			assert self.channels == self.out_channels
			conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

		# TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
		if name == "conv":
			self.Conv2d_0 = conv
			self.conv = conv
		elif name == "Conv2d_0":
			self.conv = conv
		else:
			self.conv = conv

	def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
		if self.norm is not None:
			hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

		if self.use_conv and self.padding == 0:
			pad = (0, 1, 0, 1)
			hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

		hidden_states = self.conv(hidden_states)

		return hidden_states


class PareUpBlock2D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		prev_output_channel: int,
		out_channels: int,
		temb_channels: int,
		resolution_idx: Optional[int] = None,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		output_scale_factor: float = 1.0,
		add_upsample: bool = True,
	):
		super().__init__()
		resnets = []

		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels

			resnets.append(
				PareResnetBlock2D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
				)
			)

		self.resnets = nn.ModuleList(resnets)

		if add_upsample:
			self.upsamplers = nn.ModuleList([PareUpsample2D(out_channels, use_conv=True, out_channels=out_channels)])
		else:
			self.upsamplers = None

		self.gradient_checkpointing = False
		self.resolution_idx = resolution_idx

	def forward(
		self,
		hidden_states: torch.FloatTensor,
		res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
		temb: Optional[torch.FloatTensor] = None,
		upsample_size: Optional[int] = None,
	) -> torch.FloatTensor:
		for resnet in self.resnets:
			# pop res hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]

			hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
			hidden_states = resnet(hidden_states, temb)

		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states, upsample_size)

		return hidden_states
	

class PareCrossAttnUpBlock2D(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		prev_output_channel: int,
		temb_channels: int,
		resolution_idx: Optional[int] = None,
		dropout: float = 0.0,
		num_layers: int = 1,
		transformer_layers_per_block: Union[int, Tuple[int]] = 1,
		resnet_eps: float = 1e-6,
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		num_attention_heads: int = 1,
		cross_attention_dim: int = 1280,
		output_scale_factor: float = 1.0,
		add_upsample: bool = True,
		dual_cross_attention: bool = False,
		use_linear_projection: bool = False,
		only_cross_attention: bool = False,
		upcast_attention: bool = False,
	):
		super().__init__()
		resnets = []
		attentions = []

		self.has_cross_attention = True
		self.num_attention_heads = num_attention_heads

		if isinstance(transformer_layers_per_block, int):
			transformer_layers_per_block = [transformer_layers_per_block] * num_layers

		for i in range(num_layers):
			res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
			resnet_in_channels = prev_output_channel if i == 0 else out_channels

			resnets.append(
				PareResnetBlock2D(
					in_channels=resnet_in_channels + res_skip_channels,
					out_channels=out_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,

				)
			)
			if not dual_cross_attention:
				attentions.append(
					PareTransformer2DModel(
						num_attention_heads,
						out_channels // num_attention_heads,
						in_channels=out_channels,
						num_layers=transformer_layers_per_block[i],
						cross_attention_dim=cross_attention_dim,
						norm_num_groups=resnet_groups,
						use_linear_projection=use_linear_projection,
						only_cross_attention=only_cross_attention,
						upcast_attention=upcast_attention,
					)
				)
		self.attentions = nn.ModuleList(attentions)
		self.resnets = nn.ModuleList(resnets)

		if add_upsample:
			self.upsamplers = nn.ModuleList([PareUpsample2D(out_channels, use_conv=True, out_channels=out_channels)])
		else:
			self.upsamplers = None

		self.gradient_checkpointing = False
		self.resolution_idx = resolution_idx

	def forward(
		self,
		hidden_states: torch.FloatTensor,
		res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
		temb: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		upsample_size: Optional[int] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
	) -> torch.FloatTensor:
		for resnet, attn in zip(self.resnets, self.attentions):
			# pop res hidden states
			res_hidden_states = res_hidden_states_tuple[-1]
			res_hidden_states_tuple = res_hidden_states_tuple[:-1]

			hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

			hidden_states = resnet(hidden_states, temb)
			hidden_states = attn(
				hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				cross_attention_kwargs=cross_attention_kwargs,
				attention_mask=attention_mask,
				encoder_attention_mask=encoder_attention_mask,
				return_dict=False,
			)[0]

		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states, upsample_size)

		return hidden_states


class PareUpsample2D(nn.Module):
	def __init__(
		self,
		channels: int,
		use_conv: bool = False,
		use_conv_transpose: bool = False,
		out_channels: Optional[int] = None,
		name: str = "conv",
		kernel_size: Optional[int] = None,
		padding=1,
		bias=True,
		interpolate=True,
	):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.use_conv_transpose = use_conv_transpose
		self.name = name
		self.interpolate = interpolate
		conv_cls = nn.Conv2d

		self.norm = None
		
		conv = None
		if use_conv_transpose:
			if kernel_size is None:
				kernel_size = 4
			conv = nn.ConvTranspose2d(
				channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
			)
		elif use_conv:
			if kernel_size is None:
				kernel_size = 3
			conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

		# TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
		if name == "conv":
			self.conv = conv
		else:
			self.Conv2d_0 = conv

	def forward(
		self,
		hidden_states: torch.FloatTensor,
		output_size: Optional[int] = None,
	) -> torch.FloatTensor:
		assert hidden_states.shape[1] == self.channels

		if self.norm is not None:
			hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

		if self.use_conv_transpose:
			return self.conv(hidden_states)

		dtype = hidden_states.dtype
		if dtype == torch.bfloat16:
			hidden_states = hidden_states.to(torch.float32)

		if hidden_states.shape[0] >= 64:
			hidden_states = hidden_states.contiguous()

		# if `output_size` is passed we force the interpolation output
		# size and do not make use of `scale_factor=2`
		if self.interpolate:
			if output_size is None:
				hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
			else:
				hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

		if dtype == torch.bfloat16:
			hidden_states = hidden_states.to(dtype)

		if self.use_conv:
			if self.name == "conv":
				hidden_states = self.conv(hidden_states)
			else:
				hidden_states = self.Conv2d_0(hidden_states)

		return hidden_states
	