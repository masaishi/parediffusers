# PareDiffusers
The library `pared` down the features of `diffusers` implemented the minimum function to generate images without using [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main) to understand the inner workings of the library.


## Why PareDiffusers?
PareDiffusers was born out of a curiosity and a desire to demystify the processes of generating images by diffusion models and the workings of the diffusers library.

I will write blog-style [notebooks](./notebooks) understanding how works using a top-down approach. First, generate images using diffusers to understand the overall flow, then gradually replace code with Pytorch code. In the end, we will write the code for the [PareDiffusers code](./src/parediffusers) that does not include diffusers code.

I hope that it helps others who share a similar interest in the inner workings of image generation.

## Table of Contents
### [Ch0.1 PareDiffusersPipeline](./notebooks/ch0.1_PareDiffusionPipeline.ipynb)
- [x] Generate images using diffusers
- [x] Without StableDiffusionPipeline
- [ ] Without DDIMScheduler
- [ ] Without UNet2DConditionModel
- [ ] Without AutoencoderKL
### [Ch0.1.1 Test PareDiffusersPipeline](./notebooks/ch0.1.1_TestPareDiffusionPipeline.ipynb)
- Test PareDiffusersPipeline by pip install .

## Contribution
I am starting this project to help me understand the code in order to participate in diffusers' OSS. So, I think there may be some mistakes in my explanation, so if you find any, please feel free to correct them via an issue or pull request.