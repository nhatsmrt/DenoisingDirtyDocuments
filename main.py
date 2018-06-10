import numpy as np
from pathlib import Path
import cv2
import os

from Source import DenoisingNet, LinearRegressor,\
    deflatten, threshold
from sklearn.linear_model import LinearRegression

path = Path()
d = path.resolve()
train_images_path = str(d) + "/Data/train/"
train_images_cleaned_path = str(d) + "/Data/train_cleaned/"
test_path = str(d) + "/Data/test/"
predictions_path = str(d) + "/Predictions/"


X_train = np.empty([])
y_train = []
X_test = []

image_width = 420
image_height = 540
thres = 0.65
num_epoch = 20

for filename in os.listdir(train_images_path):
    image_path = train_images_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    if img.shape[0] < 420:
        n_pad = (420 - img.shape[0]) // 2
        img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
    # small = cv2.resize(img, dsize = (image_width, image_height), interpolation = cv2.INTER_AREA) / 255
    X_train.append(img)


for filename in os.listdir(train_images_cleaned_path):
    image_path = train_images_cleaned_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    if img.shape[0] < 420:
        n_pad = (420 - img.shape[0]) // 2
        img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
    # small = cv2.resize(img, dsize = (image_width, image_height), interpolation = cv2.INTER_AREA) / 255
    y_train.append(img)

for filename in os.listdir(test_path):
    image_path = test_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    if img.shape[0] < 420:
        n_pad = (420 - img.shape[0]) // 2
        img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
    # small = cv2.resize(img, dsize=(image_width, image_height), interpolation=cv2.INTER_AREA) / 255
    X_test.append(img)


X_train = np.array(X_train).reshape((144, image_width, image_height, 1))
X_train_flat = X_train.reshape((144, -1))
y_train = np.array(y_train)
y_train_flat = y_train.reshape(144, -1)
X_test = np.array(X_test).reshape((72, image_width, image_height, 1))
X_test_flat = X_test.reshape((72, -1))


X_test_original = np.round((X_test.reshape(72, image_width, image_height) * 255))

# X_train[0] = X_train[0] * 255
# print(X_train[0].reshape(image_width, image_height).shape)
# threshold(X_train, 0.25)
# for index in range(X_train.shape[0]):
#     cv2.imwrite(predictions_path + str(index) + "_demo.png", X_train[index] * 255)
#     cv2.imwrite(predictions_path + str(index) + "_original.png", y_train[index] * 255)

# cv2.imwrite(predictions_path + "demo.png", X_train[0].reshape(image_width, image_height) * 255)
# cv2.imwrite(predictions_path + "demo_1.png", X_train[1].reshape(image_width, image_height) * 255)

# model = DenoisingNet(inp_w = image_width, inp_h = image_height, threshold = thres)
# model.fit(X_train, y_train_flat, num_epoch = num_epoch, batch_size = 16)
#
# model = LinearRegressor(inp_w = image_width, inp_h = image_height)
# model.fit(X_train, y_train_flat, num_epoch = num_epoch, batch_size = 16)

model = LinearRegression()
model.fit(X_train_flat, y_train_flat)

predictions = model.predict(X_test_flat)
threshold(predictions, thres)
predictions_deflat = np.round((deflatten(predictions, inp_w = image_width, inp_h = image_height) * 255))

for index in range(predictions_deflat.shape[0]):
    cv2.imwrite(predictions_path + str(index) + ".png", predictions_deflat[index])
    cv2.imwrite(predictions_path + str(index) + "_original.png", X_test_original[index])

