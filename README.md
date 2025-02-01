# Probabilistic PCA

Probabilistic Principal Component Analysis (PPCA) is a statistical technique used to reduce the dimensionality of data while accounting for uncertainty and noise. It extends the traditional Principal Component Analysis (PCA) by incorporating a probabilistic model, which allows for a more robust and flexible analysis of the data.

This project implements Probabilistic PCA (PPCA) for reconstructing images with missing data. It includes:
1. A **standard PPCA implementation** for general use cases.
2. An **optimized GPU-accelerated PPCA implementation** for efficient data imputation.
The PPCA model was tested for reconstructing images and time series with missing data at scale. While PPCA achieves strong quantitative results (RMSE=0.082, SSIM=0.891), visual quality can be improved with advanced techniques like deep learning.

## Results
For gray scale image reconstruction tasks:
  - **Quantitative Metrics**: PPCA outperforms baseline mean imputation (RMSE=0.158, SSIM=0.717).
  - **Visual Quality**: Reconstructions lack fine details. Future work will explore deep learning-based methods for improved perceptual quality.


## Usage
### Clone Repository
1. Clone the repository to your local machine
   
### Prerequisites  
2. Install the required dependencies by running:  
   ```bash
   pip install -r requirements.txt
   ```

### Running the Standard PPCA Model  
3. To execute the standard Probabilistic Principal Component Analysis (PPCA) implementation, run:  
   ```bash
   python ppca_standard.py
   ```
   - This script applies PPCA on a sample dataset and extracts the principal components.

### Running the Optimized PPCA Model  
4. To run the optimized PPCA version, execute:  
   ```bash
   python ppca_optimized.py
   ```
   - This version is designed for enhanced computational efficiency, potentially handling larger datasets more effectively.

### Running Tests  
5. **Test the Standard PPCA Implementation:**  
   - Open and execute `test_ppca_standard.ipynb` in Jupyter Notebook:
     ```bash
     jupyter notebook test_ppca_standard.ipynb
     ```
   - This notebook contains step-by-step evaluations of the standard PPCA model.

6. **Test the Optimized PPCA Implementation:**  
   - Open and execute `test_ppca_optimized.ipynb` in Jupyter Notebook:
     ```bash
     jupyter notebook test_ppca_optimized.ipynb
     ```
   - This notebook validates the optimized version and provides performance insights.

## Future Directions:
- Explore deep learning models (e.g., autoencoders, GANs) for better visual quality.

