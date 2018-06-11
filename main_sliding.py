import numpy as np
from pathlib import Path
import cv2
import os

from Source import DenoisingNet, MiniDenoisingNet, LinearRegressor,\
    deflatten, threshold, threshold_v2,\
    crop, slide,\
    reconstruct, reconstruct_sliding,\
    write_results

path = Path()
d = path.resolve()
train_images_path = str(d) + "/Data/train/"
train_images_cleaned_path = str(d) + "/Data/train_cleaned/"
test_path = str(d) + "/Data/test/"
predictions_path = str(d) + "/Predictions/"
sample_path = predictions_path + "sampleSubmission.csv"
demo_path = predictions_path + "demo.csv"
weight_save_path = str(d) + "/weights/model.ckpt"
weight_load_path = str(d) + "/weights/2/model.ckpt"


X_train = []
y_train = []
X_test = []

image_width = 420
image_height = 540
mini_img_width = 64
mini_img_height = 64
stride = 16

num_epoch = 10
thres = 0.75


for filename in os.listdir(train_images_path):
    image_path = train_images_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    slide(img, X_train, mini_height = mini_img_height, mini_width = mini_img_width, strides = stride)


    image_path_y = train_images_cleaned_path + filename
    img_y = cv2.imread(image_path_y, cv2.IMREAD_GRAYSCALE) / 255
    slide(img_y, y_train, mini_height = mini_img_height, mini_width = mini_img_width, strides = stride)



sub_ind = []
n_subimages = []
image_sizes = []
file_indices = []

for filename in os.listdir(test_path):
    ind = str(filename[:-4])
    file_indices.append(ind)
    image_path = test_path + filename
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    n_image, indices = slide(img, X_test, mini_height = mini_img_height,
                             mini_width = mini_img_width, strides = stride, reconstructed = True)
    image_sizes.append([img.shape[0], img.shape[1]])
    n_subimages.append(n_image)
    sub_ind.append(indices)

print("Finish Sliding")

X_train = np.array(X_train).reshape(-1, mini_img_width, mini_img_height, 1)
y_train = np.array(y_train).reshape(-1, mini_img_width * mini_img_height)
X_test = np.array(X_test).reshape(-1, mini_img_width, mini_img_height, 1)



model = MiniDenoisingNet(inp_w = mini_img_width, inp_h = mini_img_height)
model.fit(X_train, y_train, num_epoch = num_epoch,
          weight_load_path = weight_load_path,
          weight_save_path = weight_save_path, print_every = 100)

predictions = model.predict(X_test)
predictions_reconstructed = reconstruct_sliding(predictions.reshape(-1, mini_img_width, mini_img_height),
                                           image_sizes = image_sizes,
                                           ind_list = sub_ind,
                                           n_subimages = n_subimages)
predictions_thresholded = threshold_v2(predictions_reconstructed, threshold = thres)
X_test_reconstructed = reconstruct_sliding(X_test.reshape(-1, mini_img_width, mini_img_height),
                                           image_sizes = image_sizes,
                                           ind_list = sub_ind,
                                           n_subimages = n_subimages)

print("Finish reconstructing")

for ind in range(len(predictions_reconstructed)):
    cv2.imwrite(predictions_path + "_slided_predicted_" + str(file_indices[ind]) + ".png", predictions_reconstructed[ind] * 255)
    cv2.imwrite(predictions_path + "_slided_original_" + str(file_indices[ind]) + ".png", X_test_reconstructed[ind] * 255)
    cv2.imwrite(predictions_path + "_slided_thresholded_" + str(file_indices[ind]) + ".png", predictions_thresholded[ind] * 255)


