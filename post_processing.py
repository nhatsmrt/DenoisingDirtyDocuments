import numpy as np
from pathlib import Path
import cv2
import os

from Source import DenoisingNet, MiniDenoisingNet, LinearRegressor,\
    deflatten, threshold, threshold_v2, threshold_v3, crop, reconstruct
from sklearn.linear_model import LinearRegression

path = Path()
d = path.resolve()
predictions_path = str(d) + "/Predictions/"
predicted_path = predictions_path + "_slided_predicted_"

test_path = str(d) + "/Data/test/"

images = []
img_ind = []
for filename in os.listdir(test_path):
    ind = filename[:-4]
    img_ind.append(ind)
    img_path = predicted_path + ind + ".png"
    # print(img_path)
    img = cv2.imread(img_path, 0)
    images.append(img)




# images_thresholded = threshold_v3(images, upper = 0.65)

# print((images[0] * 255).astype(np.int64))
# ret3,th3 = cv2.threshold(images[0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imwrite(predictions_path + "_demo_otsu" + ".png", th3)

for ind in range(len(images)):
    # blur = cv2.GaussianBlur(images[ind], (5, 5), 0)
    # ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, th = cv2.threshold(images[ind], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(images[ind], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(predictions_path + "_slided_otsu" + str(img_ind[ind]) + ".png", th)
    # cv2.imwrite(predictions_path + "_adaptiveGaussian" + str(img_ind[ind]) + ".png", th2)


# for ind in range(len(images_thresholded)):
#     cv2.imwrite(predictions_path + "_thresholded_v3_" + str(img_ind[ind]) + ".png", images_thresholded[ind] * 255)

