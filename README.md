# PareDiffusers

[![Screenshot 2024-01-17 at 10 53 49 PM](https://github.com/masaishi/parediffusers/assets/1396267/c02bd298-c894-4fb6-b17a-a72f1b736748)](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/pipeline.py)

[![parediffusers on PyPI](https://img.shields.io/pypi/v/parediffusers.svg)](https://pypi.org/project/parediffusers)

The library `pared` down the features of `diffusers` implemented the minimum function to generate images without using [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main) to understand the inner workings of the library.


## Why PareDiffusers?
PareDiffusers was born out of a curiosity and a desire to demystify the processes of generating images by diffusion models and the workings of the diffusers library.

I will write blog-style [notebooks](./notebooks) understanding how works using a top-down approach. First, generate images using diffusers to understand the overall flow, then gradually replace code with Pytorch code. In the end, we will write the code for the [PareDiffusers code](./src/parediffusers) that does not include diffusers code.

I hope that it helps others who share a similar interest in the inner workings of image generation.

## Versions
- v0.0.0: After Ch0.0.0, inprement StableDiffusionPipeline.
- v0.1.2: After Ch0.1.0, imprement DDIMScheduler.
- v0.2.0: After Ch0.2.0, imprement UNet2DConditionModel.
- v0.3.1: After Ch0.3.0, imprement AutoencoderKL.

## Table of Contents
### [Ch0.0.0 PareDiffusersPipeline](./notebooks/ch0.0.0_ParedDiffusionPipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.0.0_ParedDiffusionPipeline.ipynb)
version: v0.0.0
- [x] Generate images using diffusers
- [x] Imprement StableDiffusionPipeline
- [ ] Imprement DDIMScheduler
- [ ] Imprement UNet2DConditionModel
- [ ] Imprement AutoencoderKL
### [Ch0.0.1 Test parediffusers](./notebooks/ch0.0.1_Test_parediffusers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.0.1_Test_parediffusers.ipynb)
- Test PareDiffusersPipeline by pip install parediffusers.
### [Ch0.0.2 Play prompt_embeds](./notebooks/ch0.0.2_Play_prompt_embeds.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.0.2_Play_prompt_embeds.ipynb)
- Play prompt_embeds, make gradation images by using two prompts.
### [Ch0.1.0: PareDDIMScheduler](./notebooks/ch0.1.0_PareDDIMScheduler.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.1.0_PareDDIMScheduler.ipynb)
version: v0.1.3
- [x] Imprement images using diffusers
- [x] Imprement StableDiffusionPipeline
- [x] Imprement DDIMScheduler
- [ ] Imprement UNet2DConditionModel
- [ ] Imprement AutoencoderKL
### [Ch0.1.1: Test parediffusers](./notebooks/ch0.1.1_Test_parediffusers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.1.1_Test_parediffusers.ipynb)
- Test PareDiffusersPipeline by pip install parediffusers.
### [Ch0.2.0: PareUNet2DConditionModel](./notebooks/ch0.2.0_PareUNet2DConditionModel.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.2.0_PareUNet2DConditionModel.ipynb)
version: v0.2.0
- [x] Generate images using diffusers
- [x] Imprement StableDiffusionPipeline
- [x] Imprement DDIMScheduler
- [x] Imprement UNet2DConditionModel
- [ ] Imprement AutoencoderKL
### [Ch0.2.1: Test parediffusers](./notebooks/ch0.2.1_Test_PareDiffusersPipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.2.1_Test_PareDiffusersPipeline.ipynb)
- Test PareDiffusersPipeline by pip install parediffusers.
### [Ch0.3.0: PareAutoencoderKL](./notebooks/ch0.3.0_PareAutoencoderKL.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.3.0_PareAutoencoderKL.ipynb)
version: v0.3.1
- [x] Generate images using diffusers
- [x] Imprement StableDiffusionPipeline
- [x] Imprement DDIMScheduler
- [x] Imprement UNet2DConditionModel
- [x] Imprement AutoencoderKL
### [Ch0.3.1: Test parediffusers](./notebooks/ch0.3.1_Test_PareDiffusersPipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.3.1_Test_PareDiffusersPipeline.ipynb)
- Test PareDiffusersPipeline by pip install parediffusers.


## Usage
```python
import torch
from parediffusers import PareDiffusionPipeline

device = torch.device("cuda")
dtype = torch.float16
model_name = "stabilityai/stable-diffusion-2"

pipe = PareDiffusionPipeline.from_pretrained(model_name, device=device, dtype=dtype)
prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"
image = pipe(prompt)
display(image)
```

## Contribution
I am starting this project to help me understand the code in order to participate in diffusers' OSS. So, I think there may be some mistakes in my explanation, so if you find any, please feel free to correct them via an issue or pull request.
