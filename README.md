# SRCNN – Image Super-Resolution (Course Project)

This repository contains code and experiments for **image super-resolution** using a  
**Super-Resolution Convolutional Neural Network (SRCNN)**, implemented as part of a machine learning course project.

The goal is to study an **image-to-image deep learning task** by deliberately degrading images
(e.g. downscaling) and restoring them as close as possible to the original resolution.

---

## Method

We use **SRCNN**, a fully convolutional neural network that learns an end-to-end mapping from
low-resolution to high-resolution images.

**Original paper**  
Chao Dong et al., *Image Super-Resolution Using Deep Convolutional Networks*, 2015  
https://arxiv.org/abs/1501.00092

---

## Code Base

This project is **based on and adapted from** the following unofficial PyTorch implementation:

https://github.com/yjn870/SRCNN-pytorch

Modifications include:
- Dataset handling and preprocessing
- Training and evaluation configuration
- Experiment management for course requirements

All credit for the original implementation goes to the respective authors.

---

## Project Scope

- Image degradation via bicubic downsampling
- CNN-based super-resolution
- Quantitative evaluation (e.g. PSNR)
- Qualitative visual comparison

This repository is intended for **educational and research purposes only**.

---

## License

The original repository’s license applies.  
This project is non-commercial and created solely for academic coursework.
