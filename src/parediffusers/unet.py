import torch
from torch import nn
from typing import List, Union
import json
from huggingface_hub import hf_hub_download
from .utils import DotDict, get_activation
from .defaults import DEFAULT_UNET_CONFIG
from .models.embeddings import PareTimestepEmbedding, PareTimesteps
from .models.unet_2d_get_blocks import pare_get_down_block, pare_get_up_block
from .models.unet_2d_mid_blocks import PareUNetMidBlock2DCrossAttn


class PareUNet2DConditionModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = DotDict(DEFAULT_UNET_CONFIG)
        self.config.update(kwargs)
        self.config.only_cross_attention = [self.config.only_cross_attention] * len(
            self.config.down_block_types
        )
        self.config.num_attention_heads = (
            self.config.num_attention_heads or self.config.attention_head_dim
        )
        self._setup_model_parameters()

        self._build_input_layers()
        self._build_time_embedding()
        self._build_down_blocks()
        self._build_mid_block()
        self._build_up_blocks()
        self._build_output_layers()

    def _setup_model_parameters(self) -> None:
        if isinstance(self.config.num_attention_heads, int):
            self.config.num_attention_heads = (self.config.num_attention_heads,) * len(
                self.config.down_block_types
            )
        if isinstance(self.config.attention_head_dim, int):
            self.config.attention_head_dim = (self.config.attention_head_dim,) * len(
                self.config.down_block_types
            )
        if isinstance(self.config.cross_attention_dim, int):
            self.config.cross_attention_dim = (self.config.cross_attention_dim,) * len(
                self.config.down_block_types
            )
        if isinstance(self.config.layers_per_block, int):
            self.config.layers_per_block = [self.config.layers_per_block] * len(
                self.config.down_block_types
            )
        if isinstance(self.config.transformer_layers_per_block, int):
            self.config.transformer_layers_per_block = [
                self.config.transformer_layers_per_block
            ] * len(self.config.down_block_types)

    def _build_input_layers(self) -> None:
        conv_in_padding = (self.config.conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            self.config.in_channels,
            self.config.block_out_channels[0],
            kernel_size=self.config.conv_in_kernel,
            padding=conv_in_padding,
        )

    def _build_time_embedding(self) -> None:
        self.config.time_embed_dim = (
            self.config.time_embedding_dim or self.config.block_out_channels[0] * 4
        )
        self.time_proj = PareTimesteps(
            self.config.block_out_channels[0],
            self.config.flip_sin_to_cos,
            self.config.freq_shift,
        )
        timestep_input_dim = self.config.block_out_channels[0]

        self.time_embedding = PareTimestepEmbedding(
            timestep_input_dim,
            self.config.time_embed_dim,
            act_fn=self.config.act_fn,
            post_act_fn=self.config.timestep_post_act,
            cond_proj_dim=self.config.time_cond_proj_dim,
        )

    def _build_down_blocks(self) -> None:
        self.down_blocks = nn.ModuleList([])
        output_channel = self.config.block_out_channels[0]
        for i, down_block_type in enumerate(self.config.down_block_types):
            input_channel = output_channel
            output_channel = self.config.block_out_channels[i]
            is_final_block = i == len(self.config.block_out_channels) - 1

            down_block = pare_get_down_block(
                down_block_type,
                num_layers=self.config.layers_per_block[i],
                transformer_layers_per_block=self.config.transformer_layers_per_block[
                    i
                ],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=self.config.time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=self.config.norm_eps,
                resnet_act_fn=self.config.act_fn,
                resnet_groups=self.config.norm_num_groups,
                cross_attention_dim=self.config.cross_attention_dim[i],
                num_attention_heads=self.config.num_attention_heads[i],
                downsample_padding=self.config.downsample_padding,
                use_linear_projection=self.config.use_linear_projection,
                only_cross_attention=self.config.only_cross_attention[i],
                upcast_attention=self.config.upcast_attention,
                dropout=self.config.dropout,
            )
            self.down_blocks.append(down_block)

    def _build_mid_block(self) -> None:
        # Supports only UNetMidBlock2DCrossAttn
        if self.config.mid_block_type != "UNetMidBlock2DCrossAttn":
            raise ValueError(
                f"mid_block_type {self.config.mid_block_type} not supported"
            )

        self.mid_block = PareUNetMidBlock2DCrossAttn(
            transformer_layers_per_block=self.config.transformer_layers_per_block[-1],
            in_channels=self.config.block_out_channels[-1],
            temb_channels=self.config.time_embed_dim,
            dropout=self.config.dropout,
            resnet_eps=self.config.norm_eps,
            resnet_act_fn=self.config.act_fn,
            output_scale_factor=self.config.mid_block_scale_factor,
            cross_attention_dim=self.config.cross_attention_dim[-1],
            num_attention_heads=self.config.num_attention_heads[-1],
            resnet_groups=self.config.norm_num_groups,
            use_linear_projection=self.config.use_linear_projection,
            upcast_attention=self.config.upcast_attention,
        )

    def _build_up_blocks(self) -> None:
        self.up_blocks = nn.ModuleList([])
        self.num_upsamplers = 0

        reversed_block_out_channels = list(reversed(self.config.block_out_channels))
        reversed_num_attention_heads = list(reversed(self.config.num_attention_heads))
        reversed_layers_per_block = list(reversed(self.config.layers_per_block))
        reversed_cross_attention_dim = list(reversed(self.config.cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(self.config.transformer_layers_per_block))
            if self.config.reverse_transformer_layers_per_block is None
            else self.config.reverse_transformer_layers_per_block
        )
        self.config.only_cross_attention = list(
            reversed(self.config.only_cross_attention)
        )

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.config.up_block_types):
            is_final_block = i == len(self.config.block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(self.config.block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = pare_get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=self.config.time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=self.config.norm_eps,
                resnet_act_fn=self.config.act_fn,
                resolution_idx=i,
                resnet_groups=self.config.norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                use_linear_projection=self.config.use_linear_projection,
                only_cross_attention=self.config.only_cross_attention[i],
                upcast_attention=self.config.upcast_attention,
                dropout=self.config.dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

    def _build_output_layers(self) -> None:
        if self.config.norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=self.config.block_out_channels[0],
                num_groups=self.config.norm_num_groups,
                eps=self.config.norm_eps,
            )
            self.conv_act = get_activation(self.config.act_fn)
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (self.config.conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            self.config.block_out_channels[0],
            self.config.out_channels,
            kernel_size=self.config.conv_out_kernel,
            padding=conv_out_padding,
        )

    @classmethod
    def _get_config(
        cls, model_name: str, filename: str = "config.json", subfolder: str = "unet"
    ) -> dict:
        config_file = hf_hub_download(
            model_name,
            filename=filename,
            subfolder=subfolder,
        )
        with open(config_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config = json.loads(text)
        return config

    @classmethod
    def _load_state_dict_into_model(
        cls, model: nn.Module, state_dict: dict
    ) -> List[str]:
        state_dict = state_dict.copy()
        error_msgs = []

        def load(module: torch.nn.Module, prefix: str = ""):
            args = (state_dict, prefix, {}, True, [], [], error_msgs)
            module._load_from_state_dict(*args)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model)
        return model

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> nn.Module:
        subfolder = kwargs.pop("subfolder", "unet")
        config_filename = kwargs.pop("config_filename", "config.json")
        model_filename = kwargs.pop(
            "model_filename", "diffusion_pytorch_model.fp16.bin"
        )

        config = cls._get_config(
            model_name, filename=config_filename, subfolder=subfolder
        )
        model_file = hf_hub_download(
            model_name,
            filename=model_filename,
            subfolder=subfolder,
        )
        state_dict = torch.load(model_file, map_location="cpu")

        model = PareUNet2DConditionModel(**config)
        model = cls._load_state_dict_into_model(model, state_dict)
        model.eval()
        return model

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        forward_upsample_size = False
        upsample_size = None

        # 1. time
        timesteps = timestep
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, None)
        aug_emb = None

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if (
                hasattr(self.mid_block, "has_cross_attention")
                and self.mid_block.has_cross_attention
            ):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample
