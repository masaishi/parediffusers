import torch
from torchvision.transforms import ToPILImage
from transformers import CLIPTokenizer, CLIPTextModel
from .scheduler import PareDDIMScheduler
from .unet import PareUNet2DConditionModel
from .vae import PareAutoencoderKL


class PareDiffusionPipeline:
    def __init__(
        self,
        tokenizer,
        text_encoder,
        scheduler,
        unet,
        vae,
        device=torch.device("cuda"),
        dtype=torch.float16,
    ):
        """
        Initialize the diffusion pipeline components.
        """
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device=device, dtype=dtype)
        self.scheduler = scheduler
        self.unet = unet.to(device=device, dtype=dtype)
        self.vae = vae.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls, model_name, device=torch.device("cuda"), dtype=torch.float16
    ):
        """
        Load all necessary components from the pretrained model.

        Args:
                model_name (str): The name of the pretrained model.
                device (torch.device, optional): The device to use for the pipeline. Defaults to torch.device("cuda").
                dtype (torch.dtype, optional): The dtype to use for the pipeline. Defaults to torch.float16.

        Returns:
                PareDiffusionPipeline: The initialized pipeline.
        """
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        )
        scheduler = PareDDIMScheduler.from_config(model_name, subfolder="scheduler")
        unet = PareUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        vae = PareAutoencoderKL.from_pretrained(model_name, subfolder="vae")
        return cls(tokenizer, text_encoder, scheduler, unet, vae, device, dtype)

    def encode_prompt(self, prompt: str):
        """
        Encode the text prompt into embeddings using the text encoder.
        """
        prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
        negative_prompt_embeds = self.get_embes([""], prompt_embeds.shape[1])
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def get_embes(self, prompt, max_length):
        """
        Encode the text prompt into embeddings using the text encoder.
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0].to(
            dtype=self.dtype, device=self.device
        )
        return prompt_embeds

    def get_latent(self, width: int, height: int):
        """
        Generate a random initial latent tensor to start the diffusion process.
        """
        return torch.randn((4, height // 8, width // 8)).to(
            device=self.device, dtype=self.dtype
        )

    def retrieve_timesteps(self, num_inference_steps=None):
        """
        Retrieve the timesteps for the diffusion process from the scheduler.
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        return timesteps, num_inference_steps

    @torch.no_grad()
    def denoise(
        self, latents, prompt_embeds, num_inference_steps=50, guidance_scale=7.5
    ):
        """
        Iteratively denoise the latent space using the diffusion model to produce an image.
        """
        timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps)

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)

            # Predict the noise residual for the current timestep
            noise_residual = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds
            )
            uncond_residual, text_cond_residual = noise_residual.chunk(2)
            guided_noise_residual = uncond_residual + guidance_scale * (
                text_cond_residual - uncond_residual
            )

            # Update latents by reversing the diffusion process for the current timestep
            latents = self.scheduler.step(guided_noise_residual, t, latents)[0]

        return latents

    def denormalize(self, image):
        """
        Denormalize the image tensor to the range [0, 255].
        """
        return (image / 2 + 0.5).clamp(0, 1)

    def tensor_to_image(self, tensor):
        """
        Convert a tensor to a PIL Image.
        """
        return ToPILImage()(tensor.detach().cpu())

    @torch.no_grad()
    def vae_decode(self, latents):
        """
        Decode the latent tensors using the VAE to produce an image.
        """
        image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
        image = self.denormalize(image)
        image = self.tensor_to_image(image)
        return image

    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: int = 7.5,
    ):
        """
        Generate an image from a text prompt using the entire pipeline.

        Args:
                prompt (str): The text prompt to generate an image from.
                height (int, optional): The height of the generated image. Defaults to 512.
                width (int, optional): The width of the generated image. Defaults to 512.
                num_inference_steps (int, optional): The number of diffusion steps to perform. Defaults to 50.
                guidance_scale (int, optional): The scale of the guidance. Defaults to 7.5.

        Returns:
                PIL.Image: The generated image.
        """
        prompt_embeds = self.encode_prompt(prompt)
        latents = self.get_latent(width, height).unsqueeze(dim=0)
        latents = self.denoise(
            latents, prompt_embeds, num_inference_steps, guidance_scale
        )
        image = self.vae_decode(latents)
        return image


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float16
    model_name = "stabilityai/stable-diffusion-2"

    pare_pipe = PareDiffusionPipeline.from_pretrained(model_name)

    prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"
    image = pare_pipe(prompt)
    image.show()
