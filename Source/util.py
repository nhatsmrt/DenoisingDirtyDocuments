import numpy as np
import pandas as pd
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

def slide(image, store, mini_width, mini_height, strides = 16, reconstructed = False):
    position = []
    n_image = 0
    for r in range(0, image.shape[0] -  mini_width + 1,  strides):
        for c in range(0, image.shape[1] -  mini_height + 1, strides):
            store.append(image[r : r + mini_width, c : c + mini_height])
            position.append([r, c])
            n_image += 1
            # print(image[r : r + mini_width, c : c + mini_height].shape)

    if (image.shape[0] -  mini_width) % strides != 0:
        for c in range(0, image.shape[1] - mini_height + 1):
            store.append(image[image.shape[0] -  mini_width : image.shape[0], c : c + mini_height])
            position.append([image.shape[0] -  mini_width, c])
            n_image += 1

    if (image.shape[1] -  mini_height) % strides != 0:
        for r in range(0, image.shape[0] - mini_width + 1):
            store.append(image[r : r + mini_width, image.shape[1] -  mini_height : image.shape[1]])
            position.append([r, image.shape[1] -  mini_height])
            n_image += 1


    if reconstructed:
        return n_image, position


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

def reconstruct_sliding(images, image_sizes, ind_list,
                        n_subimages, mini_width = 64, mini_height = 64):
    done = 0
    reconstructed_images = []
    for img_ind in range(len(n_subimages)):
        reconstructed_image = np.zeros(shape = (image_sizes[img_ind][0], image_sizes[img_ind][1]))
        mask = np.zeros(shape = (image_sizes[img_ind][0], image_sizes[img_ind][1]))
        for subimg_ind in range(n_subimages[img_ind]):
            r = ind_list[img_ind][subimg_ind][0]
            c = ind_list[img_ind][subimg_ind][1]
            reconstructed_image[r:r + mini_width, c:c + mini_height] = (reconstructed_image[r : r+mini_width, c : c+mini_height]
                                                                    * mask[r:r+mini_width, c:c+mini_height]
                                                                    + images[done]) / (mask[r:r+mini_width, c:c+mini_height] + 1)
            mask[r:r + mini_width, c:c + mini_height] = mask[r:r+mini_width, c:c+mini_height] + 1
            done += 1


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

def threshold_v3(images, lower = 0, upper = 1):
    img_copy = copy.deepcopy(images)
    for ind in range(len(img_copy)):
        img_copy[ind][np.where(img_copy[ind] > upper)] = 1
        img_copy[ind][np.where(img_copy[ind] < lower)] = 0

    return img_copy


def write_results(images, file_indices, sample_path, result_path):
    df = pd.read_csv(sample_path)
    for ind in range(len(images)):
        for r in range(images[ind].shape[0]):
            for c in range(images[ind].shape[1]):
                id = str(file_indices[ind]) + "_" + str(r + 1) + "_" + str(c + 1)
                value = images[ind][r][c]
                df.loc[df['id'] == id, 'value'] = value
                print(df.loc[df['id'] == id, 'value'])

    df.to_csv(result_path, sep=',', encoding='utf-8', index=False)
