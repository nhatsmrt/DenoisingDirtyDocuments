# Denoising Dirty Documents
## Introduction:
This project is based on Kaggle's competition: https://www.kaggle.com/c/denoising-dirty-documents
The challenge is to removed different types of synthetic noises from scanned texts.
## Approach:
Small windows (e.g ![equation](http://latex.codecogs.com/gif.latex?32%20%5Ctimes%2032)) of the scanned texts are passed through an autoencoder-like neural network. Detailed architecture can be found in code and project report.
## Some demo (from competition's test files):
Before:

![Before](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_original_136.png) 

After:

![After](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_predicted_136.png)
