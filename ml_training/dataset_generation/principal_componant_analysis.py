# -*- coding: utf-8 -*-
## This file contain function required to generate segmentation mask and dataset for training
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermography videos)

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from thermograms.Utilities import Utilities
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from PIL import Image
import json
from skimage.draw import polygon
import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip,RandomTranslation,RandomRotation
def principal_componant_analysis(data_set, no_EOF=1, plot_disp=False):
    """
    PCA convert higher dimension data to lower dimension retaining important characteristics

    Args:
        data_set (numpy array): input dataset in 3D
        no_EOF (int, optional): Number of orthogonal function to be extracted from PCA. Defaults to 1.
        plot_disp (bool, optional): To plot EOF. Defaults to False.

    Returns:
        _type_: Effective orthognal function or characterics
    """
    # checking if the dataset is a numpy array or not
    if type(data_set) != 'ndarray':
        data_set = np.array(data_set)
    if len(data_set.shape) == 3:
        (m, n, t) = data_set.shape
        data_reshaped = data_set.reshape((m * n, t))
    else:
        (m, n) = data_set.shape
    # performing PCA based on equ. 5.3.4 to 5.3.6 using numpy module    
    pca = PCA(n_components=int(no_EOF))
    charactersitic_data = pca.fit_transform(data_reshaped)
    # reshaping the  EOF to input thermogram shape
    EOFs = charactersitic_data.reshape((m, n, no_EOF))
    if plot_disp:
        img1 = Image.fromarray(EOFs[:, :, 0].astype(np.int8))
        img2 = Image.fromarray(EOFs[:, :, 0].astype(np.float32))
        plt.imshow(img1)
    return EOFs


def segmentation_mask(annotaion_path, experiment_name, height=256, width=256, linesToSkip=1):
    """
    Thsi function generates segmenation mask from labelMe json dataset
    Args:
        annotaion_path (str): Segmentation mask path 
        experiment_name (list): list of experiments
        height (int, optional): The shape of the segmentation mask. Defaults to 256.
        width (int, optional): The shape of the segmentation mask. Defaults to 256.
        linesToSkip (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_  
    """
    # importing json file which contains segmentation details
    file_name = experiment_name + '.json'
    file_path = os.path.join(annotaion_path, file_name)
    with open(file_path, 'r') as file:
        annotation = json.load(file)
        # json_key = experiment_name[:-1]+'Ã‚'+experiment_name[-1:]
    experimental_data = annotation[experiment_name]
    ## identifying the region of the features
    region_data = experimental_data['regions']
    objects = list(region_data.keys())
    object_0 = region_data['0']
    annotates = sorted(list(object_0['region_attributes']))
    # print(annotates)
    # Number of features in the segmentation json file
    no_of_classes = len(annotates)
    masks = np.zeros((height, width, no_of_classes))
    # Based on the type of the feature, the annoatation operation is performed
    for i in range(len(objects)):
        object_mask_data = region_data[str(i)]
        classes = object_mask_data['region_attributes']
        # annotation numbering 0-coating 1-substrate 2- damaged substrate 3- thermal band
        for (key, value) in classes.items():
            if classes[key] == '1':
                annotate_no = annotates.index(key)
                # print(annotate_no)
                break
        # extracting the feature shape (rectangle box point or polyline points)        
        object_shape = object_mask_data['shape_attributes']
        if object_shape['name'] == 'rect':
            # creating a rectangle bounding box based on the initial x , y, width and height information
            xmin = int(object_shape['x'])
            xmax = xmin + int(object_shape['width'])
            ymin = int(object_shape['y'])
            ymax = ymin + int(object_shape['height'])
            masks[xmin:xmax, ymin:ymax, annotate_no] = 255
        elif object_shape['name'] == 'polygon':
            # creating a polygon based x and y points
            x_points = object_shape['all_points_x']
            y_points = object_shape['all_points_y']
            x, y = polygon(x_points, y_points, shape=(height, width))
            masks[x, y, annotate_no] = 255
        else:
            print('Object shape no found', object_shape['name'])
    # print(masks.shape)
    # reshaping the segmentation mask to match the thermograms
    output_masks = masks.transpose(1, 0, 2)
    # print(output_masks.shape)
    # creating a single annoatation array from the above segmentation mask
    combined_mask = np.zeros((height, width))
    for i in range(no_of_classes):
        # print(i)
        temp = output_masks[:, :, i]
        temp[temp == 255] = i + 1
        combined_mask += temp
    return output_masks, combined_mask


def coating_thickness(experiment_list, metric=0.001):
    """
    This function generates thickness estimation output data where the thickness are extracted from the experiment name

    Args:
        experiment_list (list): list of experiment in the file
        metric (float, optional): The thckness metric (nano meter to milli meter conversion). Defaults to 0.001.

    Returns:
        _type_: output thickness dataset
    """
    # continer to store thickness 
    experiment_thickness = []
    # loop to run for all experiments
    for experiment in experiment_list.values():
        index = 0
        thickness = 0
        # runs recussively to identify all thickness provided in the experiment name
        while True:
            # identifying the index where the thickness is provided in the experiment name
            index = experiment.find("Âµm", index + 1)
            if index == -1:
                break
            thickness += int(experiment[index - 2:index]) * metric
        print(experiment, ':', thickness)
        experiment_thickness.append(thickness)
    return experiment_thickness


def Augment(tar_shape=(256,256), seed=37):
    """
    Performs data agumentation to generate more complex dataset

    Args:
        tar_shape (tuple, optional): _description_. Defaults to (256,256).
        seed (int, optional): to replicate the randomness for each run. Defaults to 37.

    Returns:
        _type_: augmented dataset
    """
    ## creating a input layer which takes thermograms and segmentation
    img = tf.keras.Input(shape=(None,None))
    msk = tf.keras.Input(shape=(None,None))
    # this layer randomlly flip the input layer
    i = tf.keras.layers.RandomFlip(seed=seed)(img)
    m = tf.keras.layers.RandomFlip(seed=seed)(msk)
    # this layer randomlly translates the input layer
    #i = tf.keras.layers.RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(i)
    #m = tf.keras.layers.RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(m)
    # this layer randomlly rotates the input layer
    #i = tf.keras.layers.RandomRotation((-0.35, 0.35),seed=seed)(i)
    #m = tf.keras.layers.RandomRotation((-0.35, 0.35),seed=seed)(m)
    return tf.keras.Model(inputs=(img,msk), outputs=(i,m))
Augment = Augment()

def complex_augment(tar_shape=(256,256), seed=37,aug_types=1):
    """ Perfomed complex sequence of augumentayion

    Args:
        tar_shape (tuple, optional): _description_. Defaults to (256,256).
        seed (int, optional): to replicate the randomness for each run. Defaults to 37.
        aug_types (int, optional): type of augumentation process. Defaults to 1.

    Returns:
        _type_: augumented data
    """
    # creating input layer of thermograms and segmentation masks
    img = Input(shape=(None,None))
    msk = Input(shape=(None,None))
    ## based on the type, the augumentation sequence is generated
    if aug_types==1:
        i = RandomFlip(seed=seed)(img)
        m = RandomFlip(seed=seed)(msk)
        return Model(inputs=(img,msk), outputs=(i,m))
    elif aug_types==2:
        i = RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(i)
        m = RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(m)
        return Model(inputs=(img,msk), outputs=(i,m))
    elif  aug_types==3:
        i = RandomRotation((-0.35, 0.35),seed=seed)(i)
        m = RandomRotation((-0.35, 0.35),seed=seed)(m)
        return Model(inputs=(img,msk), outputs=(i,m))
    elif  aug_types==4:
        i = RandomFlip(seed=seed)(img)
        m = RandomFlip(seed=seed)(msk)
        i = RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(i)
        m = RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(m)
        return Model(inputs=(img,msk), outputs=(i,m))
    elif  aug_types==5:
        i = RandomFlip(seed=seed)(img)
        m = RandomFlip(seed=seed)(msk)
        i = RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(i)
        m = RandomTranslation((-0.75, 0.75),(-0.75, 0.75),seed=seed)(m)
        i = RandomRotation((-0.35, 0.35),seed=seed)(i)
        m = RandomRotation((-0.35, 0.35),seed=seed)(m)
        return Model(inputs=(img,msk), outputs=(i,m))

def classifing_coating_thickness(thickness_list, classes=15, coating_start_range=0.001, coating_end_range=0.1):
    """
    performs classifcation of thickness list into different classes

    Args:
        thickness_list (_type_): list of thcknesses of all experiments
        classes (int, optional): Number of classes of thicness. Defaults to 15.
        coating_start_range (float, optional): lower limit of thickness. Defaults to 0.001.
        coating_end_range (float, optional): Upper limit of thickness. Defaults to 0.1.

    Returns:
        (numpy array): contains the thickness classes of the repective 
    """
    bins = np.linspace(coating_start_range, coating_end_range, classes + 1)
    classified_thickness = np.digitize(thickness_list, bins)
    return classified_thickness

if __name__ == '__main__':
    root_path = r'utilites/datasets'
    data_file_name = r'metal_data.hdf5'
    a = Utilities()
    thermal_data, experiment_list = a.open_file(root_path, data_file_name, True)
    # experiment_name = r'2021-05-11 - Variantenvergleich - VarioTherm Halogenlampe - Winkel 30Â°'
    # experiment_name = r'2021-05-25 - Variantenvergleich - VarioTherm Halogenlampe - Belichtungszeit 5s'
    experiment_name = r'2021-12-15-Materialstudie_Metallproben-ML1-laserbehandelte_Probe-150W-10s'
    experimental_data = thermal_data[experiment_name]
    EOFs = principal_componant_analysis(experimental_data)
    mask = np.zeros(shape=np.squeeze(EOFs).shape)
    plt.imshow(np.squeeze(EOFs))
    plt.colorbar()
    plt.show()
    annotation_path = r'ml_training\dataset_generation\masks\JSON'
    output_masks, combined_mask = segmentation_mask(annotation_path, experiment_name)
    temp_mask = combined_mask
    thermal_band = np.where(temp_mask == 3)
    temp_mask[temp_mask == 1] = 1
    temp_mask[temp_mask == 2] = 1
    temp_mask[temp_mask == 3] = 0
    temp_mask[temp_mask == 0] = 0
    plt.imsave('./manual_mask.png', temp_mask, cmap='binary_r')
    mask = np.zeros(shape=np.squeeze(EOFs).shape)
    plt.imshow(np.squeeze(EOFs))
    plt.colorbar()
    plt.show()
    mask[np.squeeze(EOFs) > 200] = 1
    # mask[np.where((np.squeeze(EOFs) >= -150) & (np.squeeze(EOFs) <=-10))]=2
    # mask[np.squeeze(EOFs) <= -300]=4
    # mask[np.where((np.squeeze(EOFs) >= -300) & (np.squeeze(EOFs) <=0))]=3
    plt.imshow(np.squeeze(mask), cmap='binary_r')
    plt.show()
    plt.imsave('Documents/temp/mask.png', np.squeeze(mask), cmap='binary_r')
    img1 = cv.imread('Documents/temp/manual_mask.png', 0)
    img2 = cv.imread('Documents/temp/mask.png', 0)
    print(img2.shape, img1.shape)
    img_bwa = cv.bitwise_and(img1, img2)
    img_bwo = cv.bitwise_or(img1, img2)
    img_bno = cv.bitwise_not(img1, img2)
    img_bwx = cv.bitwise_xor(img_bwa, img2)
    img1[img1 == 255] = 1
    img2[img2 == 255] = 1
    img_bwx[img_bwx == 255] = 1
    img_bwa[img_bwa == 255] = 1
    categorical_mask = np.zeros((256, 256, 3))
    categorical_mask[:, :, 0] = img2
    categorical_mask[:, :, 1] = img_bwa
    categorical_mask[:, :, 2] = img_bwx
    print(np.unique(categorical_mask), np.unique(img_bno))
    combined_mask = np.zeros((256, 256))
    for i in range(0, 3):
        temp = categorical_mask[:, :, i]
        temp[temp == 1] = i
        combined_mask += temp
    combined_mask[combined_mask == 3] = 1
    combined_mask[combined_mask == 0] = 3
    combined_mask[combined_mask == 2] = 0
    combined_mask[combined_mask == 3] = 2
    combined_mask[thermal_band] = 3
    name = r"Documents/temp/final_masks" + r"/" + experiment_name
    np.save(name, combined_mask)
    ar = np.load(name + '.npy')
    plt.imshow(ar, cmap='gray')
    plt.colorbar()
    plt.show()
