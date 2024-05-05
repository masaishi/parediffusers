import json
from typing import List
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from .utils import DotDict
from .defaults import DEFAULT_VAE_CONFIG
from .models.vae_blocks import (
    PareEncoder,
    PareDecoder,
    PareDiagonalGaussianDistribution,
)


class PareAutoencoderKL(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = DotDict(DEFAULT_VAE_CONFIG)
        self.config.update(kwargs)

        # pass init params to Encoder
        self.encoder = PareEncoder(
            in_channels=self.config.in_channels,
            out_channels=self.config.latent_channels,
            down_block_types=self.config.down_block_types,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
        )

        # pass init params to Decoder
        self.decoder = PareDecoder(
            in_channels=self.config.latent_channels,
            out_channels=self.config.out_channels,
            up_block_types=self.config.up_block_types,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
        )

        self.quant_conv = nn.Conv2d(
            2 * self.config.latent_channels, 2 * self.config.latent_channels, 1
        )
        self.post_quant_conv = nn.Conv2d(
            self.config.latent_channels, self.config.latent_channels, 1
        )

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

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
        subfolder = kwargs.pop("subfolder", "vae")
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

        model = PareAutoencoderKL(**config)
        model = cls._load_state_dict_into_model(model, state_dict)
        model.eval()
        return model

    def encode(self, x: torch.FloatTensor):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = PareDiagonalGaussianDistribution(moments)
        return posterior

    def _decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        decoded = self._decode(z)
        return decoded

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        x = sample
        posterior = self.encode(x).latent_dist
        z = posterior.mode()
        dec = self.decode(z).sample
        return dec
