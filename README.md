# reproduce_NCprior


This is the official codebase of the course project report for IFT6269, lead by Jerry Huang, Xinyu Yuan and Le Zhang.

## Overview


We reproduce the methods presented by  [Aneja et al. (2021)](https://arxiv.org/abs/2010.02917) to train
autoencoder priors for high-quality image generation. It is common for variational autoencoders (VAEs) to generate poor images when sampling from the prior without tempering. To tackle this issue, the authors proposed an energy-based prior defined
by the product of a base prior distribution and a reweighting factor, to bridge the gap between the base prior and the aggregate posterior. 

For our project, we re-implement the training of this re-weighting factor from scratch and reproduce a subset of the results presented in the original paper.

This codebase is based on PyTorch and Pythae. It supports training and inference with CPUs or GPUs.

## Installation




```bash
pip install pythae
```


## File Description

### preprocessors.py

The purpose of the preprocessor is to ensure the data is not corrupted (no nan), reshape
it in case inconsistencies are detected, normalize it and converted it to a format handled by the
`pythae.trainers.Trainer`. In particular, an input data is converted to a
`torch.Tensor` and all the data is gather into a `pythae.data.datastest.BaseDatset`
instance.

By choice, we do not provided very advanced preprocessing functions (such as image registrations)
since the augmentation method should be robust to huge differences in the data and be able to
reproduce and account for this diversity. More advanced preprocessing is up to the user.

### ncp.ipynb


This notebook provides the training of the three stages mentioned in the report on both MNIST and CIFAR10. Some visualization results are also included, say reconstructed images, generated images and interplorations of images.