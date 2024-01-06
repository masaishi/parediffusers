import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from .utils import DictDotNotation

class PareDDIMScheduler:
	def __init__(
		self,
		config_dict: dict
	):
		"""Initialize beta and alpha values for the scheduler."""
		self.config = DictDotNotation(**config_dict)
		self.betas = torch.linspace(self.config.beta_start**0.5, self.config.beta_end**0.5, self.config.num_train_timesteps, dtype=torch.float32) ** 2
		self.alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
		self.final_alpha_cumprod = torch.tensor(1.0)
	
	@classmethod
	def from_config(cls, model_name: str, subfolder: str = "scheduler", filename: str = "scheduler_config.json") -> "ParedDDIMScheduler":
		"""Create scheduler instance from configuration file."""
		config_file = hf_hub_download(
			model_name,
			filename=filename,
			subfolder=subfolder
		)
		with open(config_file, "r", encoding="utf-8") as reader:
			text = reader.read()
		config_dict = json.loads(text)
		return cls(config_dict)

	def set_timesteps(self, num_inference_steps: int, device: torch.device = None) -> None:
		"""Set the timesteps for the scheduler based on the number of inference steps."""
		self.num_inference_steps = num_inference_steps
		step_ratio = self.config.num_train_timesteps // self.num_inference_steps
		timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
		timesteps += self.config.steps_offset
		self.timesteps = torch.from_numpy(timesteps).to(device)

	def step(
		self,
		model_output: torch.FloatTensor,
		timestep: int,
		sample: torch.FloatTensor,
	) -> list:
		"""Perform a single step of denoising in the diffusion process."""
		prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

		alpha_prod_t = self.alphas_cumprod[timestep]
		alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

		beta_prod_t = 1 - alpha_prod_t
		pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
		pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

		pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
		prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

		return prev_sample, pred_original_sample