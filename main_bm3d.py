import numpy as np
import cv2
from pybm3d import bm3d
from pathlib import Path
import os

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
thres = 0.65
num_epoch = 20

for filename in os.listdir(train_images_path):
    image_path = train_images_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if img.shape[0] < 420:
    #     n_pad = (420 - img.shape[0]) // 2
    #     img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
    X_train.append(img)

    image_path_y = train_images_cleaned_path + filename
    img_y = cv2.imread(image_path_y, cv2.IMREAD_GRAYSCALE)
    # if img_y.shape[0] < 420:
    #     n_pad = (420 - img_y.shape[0]) // 2
    #     img_y = np.pad(img_y, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
    y_train.append(img_y)

index_list = []

for filename in os.listdir(test_path):
    ind = filename[:-4]
    index_list.append(ind)
    image_path = test_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if img.shape[0] < 420:
    #     n_pad = (420 - img.shape[0]) // 2
    #     img = np.pad(img, pad_width = ((n_pad, n_pad), (0, 0)), mode = 'constant', constant_values = (((0, 0), (0, 0))))
    X_test.append(img)

print("Finish reading input")
restored_demo = bm3d.bm3d(X_test[0], 40)
cv2.imwrite(predictions_path + "_bm3d_demo.png", restored_demo)
cv2.imwrite(predictions_path + "_bm3d_demo_original.png", X_test[0])