# Experiment 7 — Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and VAEs

CS3807 Deep Learning Laboratory — end-to-end study of four autoencoder variants on MNIST: a Fully Connected
Autoencoder, a Convolutional Autoencoder, a Denoising Convolutional Autoencoder, and a Variational
Autoencoder (VAE).

## Contents

- `Experiment_7.ipynb` — full notebook: data loading/preprocessing, all four models, training, all required
  plots and metric tables, the latent-dimension sweep, and the additional exercises (β-VAE, transposed-conv
  decoder, salt-and-pepper noise comparison, larger generation grid, targeted latent interpolation).
- `requirements.txt` — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Then open `Experiment_7.ipynb` in Jupyter or upload it to Google Colab (GPU runtime recommended for faster
training).

## What's implemented

| Model | Key idea | Latent size |
|---|---|---|
| Fully Connected AE | Dense encoder/decoder on flattened 784-pixel input | 16 |
| Convolutional AE | Conv2D + MaxPooling encoder, Conv2D + UpSampling2D decoder | 7×7×64 feature map |
| Denoising CAE | Same Conv-AE architecture, trained on Gaussian-noised input with clean target | 7×7×64 feature map |
| VAE | Probabilistic encoder (μ, log σ²) with reparameterization trick, trained with reconstruction + KL loss | 2 |

## Results summary

| Model | MSE | MAE | SSIM | Parameters |
|---|---|---|---|---|
| FC Autoencoder | 0.02214 | 0.05910 | 0.7519 | 211,040 |
| Convolutional AE | 0.00270 | 0.01589 | 0.9710 | 74,497 |
| Denoising CAE (σ=0.2) | 0.00451 | 0.02116 | 0.9477 | 74,497 |
| VAE | 0.04427 | 0.10253 | 0.5116 | 284,933 |

The VAE's lower reconstruction scores reflect its different objective (trading reconstruction sharpness for
a continuous, generative latent space) rather than a weaker model — see the notebook's latent-space
visualization, random-sample generation, and interpolation sections for its actual purpose.

## Dataset

MNIST handwritten digits, loaded via `tf.keras.datasets.mnist`. A fixed subset of 10,000 training and 2,000
test images is used throughout, per the lab brief's recommendation.

## Author

D Kanishkarane — B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai
