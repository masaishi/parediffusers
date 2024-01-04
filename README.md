# PareDiffusers

[![parediffusers on PyPI](https://img.shields.io/pypi/v/parediffusers.svg)](https://pypi.org/project/parediffusers)

The library `pared` down the features of `diffusers` implemented the minimum function to generate images without using [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main) to understand the inner workings of the library.


## Why PareDiffusers?
PareDiffusers was born out of a curiosity and a desire to demystify the processes of generating images by diffusion models and the workings of the diffusers library.

I will write blog-style [notebooks](./notebooks) understanding how works using a top-down approach. First, generate images using diffusers to understand the overall flow, then gradually replace code with Pytorch code. In the end, we will write the code for the [PareDiffusers code](./src/parediffusers) that does not include diffusers code.

I hope that it helps others who share a similar interest in the inner workings of image generation.

## Versions
- v0.0.0: After Ch0.0.0, without StableDiffusionPipeline.

## Table of Contents
### [Ch0.0.0 PareDiffusersPipeline](./notebooks/ch0.0.0_ParedDiffusionPipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.0.0_ParedDiffusionPipeline.ipynb)

- [x] Generate images using diffusers
- [x] Without StableDiffusionPipeline
- [ ] Without DDIMScheduler
- [ ] Without UNet2DConditionModel
- [ ] Without AutoencoderKL
### [Ch0.0.1 Test PareDiffusersPipeline](./notebooks/ch0.0.1_Test_ParedDiffusionPipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.0.1_Test_ParedDiffusionPipeline.ipynb)
- Test PareDiffusersPipeline by pip install .
### [Ch0.0.2 Play prompt_embeds](./notebooks/ch0.0.2_Play_prompt_embeds.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaishi/parediffusers/blob/main/notebooks/ch0.0.2_Play_prompt_embeds.ipynb)
- Play prompt_embeds, make gradation images by using two prompts.

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