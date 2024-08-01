# Denoising Diffusion Implicit Models (DDIMs)
- Title: Denoising Diffusion Implicit Models
- Author: Jiaming Song, Chenlin Meng & Stefano Ermon
- Conf: ICLR 2021
- URL: [Arxiv](https://arxiv.org/pdf/2010.02502)

## Summary of the Paper
The paper presents Denoising Diffusion Implicit Models (DDIMs), an advancement over Denoising Diffusion Probabilistic Models (DDPMs) for image generation. DDIMs maintain the same training procedure as DDPMs but introduce non-Markovian diffusion processes, enabling faster sampling and producing high-quality images. DDIMs can generate samples 10× to 50× faster than DDPMs, offering a trade-off between computation and sample quality. 

## The Technical Core
**Accelerated Sampling**: The paper presents methods for significantly accelerating the sampling process. By choosing appropriate sub-sequences and variance hyperparameters, the authors demonstrate that DDIMs can produce high-quality samples in far fewer steps compared to traditional DDPMs, achieving speedups of 10× to 100×.

**Innovative Methodology**: The introduction of a non-Markovian forward diffusion process and the concept of implicit probabilistic models represent significant advancements in the field of generative modeling. These innovations provide new avenues for research and application.


## What I implemented
- Both DDPM and DDIM Diffuser from scratch.
  - I defined the Diffuser class of DDPM first and implemented DDIM by inheriting that class.
- Implemented evaluation by FID.

## Setup
```
cd DDIM
pip install -r requirements.txt
DDIM.py
```

## Structure
- utils/Diffusers.py
  - This file defines DDPMDiffuser and DDIMDiffuser classes for image generation using diffusion models, including methods for noise addition, denoising, sampling, and image conversion to PIL format.
- utils/FID.py
  - This code calculates the FID score between real and generated images using Inception V3, including methods for preprocessing, obtaining activations, and computing the Fréchet distance.
- utils/lib.py
  - This code provides functions for displaying images, loading and training a model, and generating sample images with a diffuser.
- utils/UNet.py
  - This code defines a U-Net model with time-based positional encoding for generating images, including ConvBlock for convolutional layers and UNet for the full architecture.
- DDIM.py
  - This script trains or loads a U-Net model to generate images using a DDIM diffuser, evaluates the results with FID score, and saves the generated images and model weights.

## Experiments
### Device
NVIDIA RTX A3000 8GB

## Hyperparameters
- Diffusion Steps: 1000
- Epochs: 100
- Noise Schedule: $\beta_{\text{start}}=0.0001, \beta_{\text{end}}=0.02$, linear interpolation
- Learning Rate: 1e-3
- Batch Size: 128
- Model Architecture: U-Net

## Results

Compared to DDPM, DDIM can generate high-quality images with fewer sampling steps. The table shows the time required to generate 5000 images and the FID scores. Notably, DDPM takes 3 hours, 26 minutes, and 49 seconds for 1000 timesteps, whereas DDIM finishes in 7 minutes and 31 seconds for 50 timesteps. While DDPM achieves a distribution closer to the dataset in terms of FID, there were no noticeable visual differences between the two methods.

<div align="center">
  
| Step | 50 | 100 | 500 | 1000 |
| :--: | :--: | :--: | :--: | :--: |
| DDPM FID | - | - | - | 39.17 |
| DDPM time | - | - | - | 3:26:49 |
| DDIM FID | 49.83 | 48.75 | 49.32 | 46.61 |
| DDIM time | 0:07:31 | 0:19:35 | 1:29:29 | 2:24:45 |

</div>

<div style="text-align:center"><img src="./ddim_images/generated_images.png" /></div>
