# Probabilistic PCA

Probabilistic Principal Component Analysis (PPCA) is a statistical technique used to reduce the dimensionality of data while accounting for uncertainty and noise. It extends the traditional Principal Component Analysis (PCA) by incorporating a probabilistic model, which allows for a more robust and flexible analysis of the data.

This project implements Probabilistic Principal Component Analysis and extends the model for data imputation with a GPU-accelerated Probabilistic PCA (PPCA) tested for reconstructing images with missing data at scale. While PPCA achieves strong quantitative results (RMSE=0.082, SSIM=0.891), visual quality can be improved with advanced techniques like deep learning.

## Results
For gray scale image reconstruction tasks:
  - **Quantitative Metrics**: PPCA outperforms baseline mean imputation (RMSE=0.158, SSIM=0.717).
  - **Visual Quality**: Reconstructions are smooth but may lack fine details. Future work will explore deep learning-based methods for improved perceptual quality.

## Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python test.py` to reconstruct images with missing data.

## Future Work
- Explore deep learning models (e.g., autoencoders, GANs) for better visual quality.
- Extend to time series imputation.
