# Denoising Dirty Documents
## Introduction:
This project is based on Kaggle's competition: https://www.kaggle.com/c/denoising-dirty-documents
The challenge is to removed different types of synthetic noises from scanned texts.
## Approach:
Small windows (e.g [equation](http://www.sciweavers.org/tex2img.php?eq=32%20%5Ctimes%2032&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="32 \times 32" width="67" height="17) of the scanned texts are passed through an autoencoder-like neural network. Detailed architecture can be found in code and project report.
## Some demo (from competition's test files):