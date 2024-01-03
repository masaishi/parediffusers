# PareDiffusers
The library `pared` down the features of `diffusers` implemented the minimum function to generate images without using [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main) to understand the inner workings of the library.


## Why PareDiffusers?
PareDiffusers was born out of a curiosity and a desire to demystify the processes of generating images by diffusion models and the workings of the diffusers library.

I will write blog-style [notebooks](./notebooks) understanding how works using a top-down approach. First, generate images using diffusers to understand the overall flow, then gradually replace code with Pytorch code. In the end, we will write the code for the [PareDiffusers code](./src/parediffusers) that does not include diffusers code.

Write one notebook for each chapter and code for that version of the library. Each version of the code is maintained by a branch. E.g. `ch01_StableDiffusionPipeline` for [Chapter 1](./notebooks/ch01_StableDiffusionPipeline.ipynb) and branch is `ch01`.

I hope that it helps others who share a similar interest in the inner workings of image generation.

## Table of Contents
### [Ch0.1 StableDiffusionPipeline](./notebooks/ch0.1_StableDiffusionPipeline.ipynb)


## Contribution
I am starting this project to help me understand the code in order to participate in diffusers' OSS. So, I think there may be some mistakes in my explanation, so if you find any, please feel free to correct them via an issue or pull request.