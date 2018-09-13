# Denoising Dirty Documents
## Author
Nhat Pham (https://github.com/nhatsmrt) & Hoang Phan (https://github.com/petrpan26)
## Introduction
This project is based on Kaggle's competition: https://www.kaggle.com/c/denoising-dirty-documents
<br/>
The challenge is to removed different types of synthetic noises from scanned texts.
## Approach
Small windows (e.g of size ![equation](http://latex.codecogs.com/gif.latex?32%20%5Ctimes%2032)) of the scanned texts are passed through an autoencoder-like neural network.
<br/>
The network has a convolutional encoder with residual connections. For the decoder component, a simple feedforward layer is sufficient. However, a deconvolutional layer is used because it has less parameters, which speeds up training time.
<br/>
Detailed architecture can be found in code and project report.
## Some demo (from competition's test files)
### Before:
<br/>
![Before](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_original_136.png)
<br/>
### After:
<br/>
![After](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_predicted_136.png)
<br/>
### Before:
<br/>
![Before](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_original_7.png)
<br/>
### After:
<br/>
![After](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_predicted_7.png)
<br/>
### Before:
<br/>
![After](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_original_1.png)
<br/>
### After:
<br/>
![After](https://github.com/nhatsmrt/DenoisingDirtyDocuments/blob/sliding/Predictions/_slided_predicted_1.png)
<br/>
