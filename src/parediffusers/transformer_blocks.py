import torch
from torch import nn
from typing import Optional, Dict, Any
from .attension import PareAttention, PareFeedForward

class PareBasicTransformerBlock(nn.Module):
	def __init__(
		self,
		dim: int,
		num_attention_heads: int,
		attention_head_dim: int,
		dropout=0.0,
		cross_attention_dim: Optional[int] = None,
		num_embeds_ada_norm: Optional[int] = None,
		attention_bias: bool = False,
		only_cross_attention: bool = False,
		double_self_attention: bool = False,
		upcast_attention: bool = False,
		norm_elementwise_affine: bool = True,
		norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
		norm_eps: float = 1e-5,
		final_dropout: bool = False,
		ff_inner_dim: Optional[int] = None,
		ff_bias: bool = True,
		attention_out_bias: bool = True,
	):
		super().__init__()
		self.only_cross_attention = only_cross_attention

		self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
		self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
		self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
		self.use_layer_norm = norm_type == "layer_norm"
		self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

		self.pos_embed = None

		self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

		self.attn1 = PareAttention(
			query_dim=dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			dropout=dropout,
			bias=attention_bias,
			cross_attention_dim=cross_attention_dim if only_cross_attention else None,
			upcast_attention=upcast_attention,
			out_bias=attention_out_bias,
		)

		# 2. Cross-Attn
		if cross_attention_dim is not None or double_self_attention:
			self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

			self.attn2 = PareAttention(
				query_dim=dim,
				cross_attention_dim=cross_attention_dim if not double_self_attention else None,
				heads=num_attention_heads,
				dim_head=attention_head_dim,
				dropout=dropout,
				bias=attention_bias,
				upcast_attention=upcast_attention,
				out_bias=attention_out_bias,
			)
		else:
			self.norm2 = None
			self.attn2 = None

		# 3. Feed-forward
		self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

		self.ff = PareFeedForward(
			dim,
			dropout=dropout,
			final_dropout=final_dropout,
			inner_dim=ff_inner_dim,
			bias=ff_bias,
		)

		# let chunk size default to None
		self._chunk_size = None
		self._chunk_dim = 0

	def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
		# Sets chunk feed-forward
		self._chunk_size = chunk_size
		self._chunk_dim = dim

	def forward(
		self,
		hidden_states: torch.FloatTensor,
		attention_mask: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		timestep: Optional[torch.LongTensor] = None,
		cross_attention_kwargs: Dict[str, Any] = None,
		class_labels: Optional[torch.LongTensor] = None,
		added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
	) -> torch.FloatTensor:
		norm_hidden_states = self.norm1(hidden_states)
		if self.pos_embed is not None:
			norm_hidden_states = self.pos_embed(norm_hidden_states)

		# 2. Prepare GLIGEN inputs
		cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
		gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

		attn_output = self.attn1(
			norm_hidden_states,
			encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
			attention_mask=attention_mask,
			**cross_attention_kwargs,
		)
		
		hidden_states = attn_output + hidden_states
		if hidden_states.ndim == 4:
			hidden_states = hidden_states.squeeze(1)

		# 2.5 GLIGEN Control
		if gligen_kwargs is not None:
			hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

		# 3. Cross-Attention
		if self.attn2 is not None:
			if self.use_ada_layer_norm:
				norm_hidden_states = self.norm2(hidden_states, timestep)
			elif self.use_ada_layer_norm_zero or self.use_layer_norm:
				norm_hidden_states = self.norm2(hidden_states)
			elif self.use_ada_layer_norm_single:
				# For PixArt norm2 isn't applied here:
				# https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
				norm_hidden_states = hidden_states
			elif self.use_ada_layer_norm_continuous:
				norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
			else:
				raise ValueError("Incorrect norm")

			if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
				norm_hidden_states = self.pos_embed(norm_hidden_states)

			attn_output = self.attn2(
				norm_hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=encoder_attention_mask,
				**cross_attention_kwargs,
			)
			hidden_states = attn_output + hidden_states

		# 4. Feed-forward
		if self.use_ada_layer_norm_continuous:
			norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
		elif not self.use_ada_layer_norm_single:
			norm_hidden_states = self.norm3(hidden_states)

		ff_output = self.ff(norm_hidden_states)
		hidden_states = ff_output + hidden_states
		if hidden_states.ndim == 4:
			hidden_states = hidden_states.squeeze(1)

		return hidden_states