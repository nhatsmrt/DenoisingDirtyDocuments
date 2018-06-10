import numpy as np
from pathlib import Path
import cv2
import os

from Source import DenoisingNet, MiniDenoisingNet, LinearRegressor,\
    deflatten, threshold, threshold_v2, crop, reconstruct
from sklearn.linear_model import LinearRegression

path = Path()
d = path.resolve()
train_images_path = str(d) + "/Data/train/"
train_images_cleaned_path = str(d) + "/Data/train_cleaned/"
test_path = str(d) + "/Data/test/"
predictions_path = str(d) + "/Predictions/"


X_train = []
y_train = []
X_test = []

image_width = 420
image_height = 540
mini_img_width = 30
mini_img_height = 30

num_epoch = 20
thres = 0.25


for filename in os.listdir(train_images_path):
    image_path = train_images_path + filename
    image_path_2 = train_images_cleaned_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    img_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE) / 255


    if img.shape[0] < 420:
        n_pad = 6
        img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))

    crop(img, X_train)


for filename in os.listdir(train_images_cleaned_path):
    image_path = train_images_cleaned_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    if img.shape[0] < 420:
        n_pad = 6
        img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))

    crop(img, y_train)

reconstruct_indices = []

for filename in os.listdir(test_path):
    image_path = test_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    if img.shape[0] < 420:
        n_pad = 6
        original_width = img.shape[0]
        original_height = img.shape[1]
        img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
        reconstruct_indices.append((img.shape[0] // mini_img_width, img.shape[1] // mini_img_height, original_width, original_height))
    else:
        reconstruct_indices.append((img.shape[0] // mini_img_width, img.shape[1] // mini_img_height))
    crop(img, X_test)


X_train = np.array(X_train).reshape(-1, 30, 30, 1)
y_train = np.array(y_train).reshape(-1, 30, 30, 1)
y_train_flat = y_train.reshape(31968, -1)
X_test = np.array(X_test).reshape(-1, 30, 30, 1)


model = MiniDenoisingNet(inp_w = mini_img_width, inp_h = mini_img_height)
model.fit(X_train, y_train_flat, num_epoch = num_epoch)

predictions = model.predict(X_test)
predictions_reconstructed = reconstruct(predictions.reshape(-1, 30, 30), reconstruct_indices)
predictions_thresholded = threshold_v2(predictions_reconstructed, threshold = thres)
X_test_reconstructed = reconstruct(X_test.reshape(-1, 30, 30), reconstruct_indices)

for ind in range(len(predictions_reconstructed)):
    cv2.imwrite(predictions_path + "_predicted_" + str(ind) + ".png", predictions_reconstructed[ind] * 255)
    cv2.imwrite(predictions_path + "_original_" + str(ind) + ".png", X_test_reconstructed[ind] * 255)
    cv2.imwrite(predictions_path + "_thresholded_" + str(ind) + ".png", predictions_thresholded[ind] * 255)


