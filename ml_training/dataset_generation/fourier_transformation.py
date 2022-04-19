# -*- coding: utf-8 -*-
## This file contain performs filtering of dataset based on phases
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermography videos)

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def fourier_transformation(experiment_data, scaling_type='normalization', index=1):
    """
    This function performs Fast fourier transformation on thermographic data and filters reflection phase data for object segmentation.

    Args:
        experiment_data (hd5): thermographic data 
        scaling_type (str, optional): Type of scaling . Defaults to 'normalization'.
        index (int, optional): index to remove the corrupted thermogram data . Defaults to 1.

    Returns:
        _type_:  filtered dataset,reflection phase start and end index
    """
    # converting hd5 dataset to numpy array
    data = np.array(experiment_data)
    # performing fast fourier transformation based on equ. 5.3.1
    fft_data= np.fft.fftn(data[:, :, index:-index], axes=(0, 1))
    # Calculating amplitude of the fft data based on equ. 5.3.2
    amplitude = np.abs(fft_data)
    #converting 3d data to a 2d sequence for analysis
    amplitude_sequence = np.mean(amplitude, axis=(0, 1))
    # calculating step difference to identify peak changes in amplitude
    delta_amplitude = np.diff(amplitude_sequence)

    sorted = np.sort(delta_amplitude)
    # getting index of frames which highest and lowest difference
    sorted_index = np.argsort(delta_amplitude)
    if sorted_index[-1] > sorted_index[-2]:
        reflection_start_index = sorted_index[-2]
    else:
        reflection_start_index = sorted_index[-1]
    # reflection_start_index = np.argmax(delta_amplitude) + 5
    # Adding tolerance to the initial frame index to correct the reflection phase 
    reflection_start_index = reflection_start_index - 5
    reflection_end_index = np.argmin(delta_amplitude) - 5
    print('reflection_start_index: ',reflection_start_index,'  reflection_end_index: ', reflection_end_index)
    # filtering relfection phase in experimental data 
    if reflection_start_index > reflection_end_index:
        filtered_data = data[:, :, 150:500 + 5]
    else:
        filtered_data = data[:, :, reflection_start_index:reflection_end_index]
    print('The size of filtered data:', filtered_data.shape)
    # performing gaussian normalization on reflection phase data
    if scaling_type == 'normalization':
        standardizing = StandardScaler()
        std_output_data = standardizing.fit_transform(
            filtered_data.reshape(filtered_data.shape[0], -1)).reshape(filtered_data.shape)
        return std_output_data,reflection_start_index,reflection_end_index
    # performing min max scaling between (0,1) on reflection phase data
    elif scaling_type == 'min_max':
        normalizing = MinMaxScaler(feature_range=(0, 1))
        nrm_output_data = normalizing.fit_transform(
            filtered_data.reshape(filtered_data.shape[0], -1)).reshape(filtered_data.shape)
        return nrm_output_data,reflection_start_index,reflection_end_index
    else:
        return filtered_data,reflection_start_index,reflection_end_index
