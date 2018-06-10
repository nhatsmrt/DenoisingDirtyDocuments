import numpy as np
import copy

def deflatten(images, inp_w = 32, inp_h = 32):
    return images.reshape((images.shape[0], inp_w, inp_h))

def crop(image, store, mini_width = 30, mini_height = 30):
    # images = []
    n_r = image.shape[0] // mini_width
    n_c = image.shape[1] // mini_height

    for r in range(n_r):
        for c in range(n_c):
            mini_image = image[r * mini_width : (r + 1) * mini_width, c * mini_height : (c+ 1) * mini_height]
            # np.append(store, mini_image, axis = 0)
            store.append(mini_image)

def reconstruct(images, indices_list, mini_width = 30, mini_height = 30):
    done = 0
    reconstructed_images = []
    for indices in indices_list:
        reconstructed_image = np.zeros(shape = (indices[0] * mini_width, indices[1] * mini_height))
        for r in range(indices[0]):
            for c in range(indices[1]):
                reconstructed_image[r * mini_width : (r + 1) * mini_width, c * mini_height : (c+ 1) * mini_height] = images[done]
                done += 1
        if len(indices) == 4:
            pad_r = (indices[0] * mini_width - indices[2]) // 2
            pad_c = (indices[1] * mini_width - indices[3]) // 2
            reconstructed_image = reconstructed_image[pad_r : indices[0] * mini_width - pad_r, pad_c : indices[1] * mini_height - pad_c]
        reconstructed_images.append(reconstructed_image)
    return reconstructed_images

def threshold(images, threshold):
    img_copy = copy.deepcopy(images)
    img_copy[np.where(images > threshold)] = 1
    img_copy[np.where(images < threshold)] = 0
    return img_copy

def threshold_v2(images, threshold):
    img_copy = copy.deepcopy(images)
    for ind in range(len(img_copy)):
        img_copy[ind][np.where(img_copy[ind] > threshold)] = 1
        img_copy[ind][np.where(img_copy[ind] < threshold)] = 0

    return img_copy


