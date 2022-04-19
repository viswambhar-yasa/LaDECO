# -*- coding: utf-8 -*-
## This file contain data evaluation methods to identify phases and thermal profile of vario therm data
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermography videos)
import os
import numpy as np
import seaborn as sns
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from thermograms.Utilities import Utilities, h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def phases_identification(experiment_data,  index=1):
    """_summary_

    Args:
        experiment_data (hd5): Thermographic data 
        index (int, optional): tolernance value for slicing the data to remove error thermograms. Defaults to 1.

    Returns:
        _type_: filtered data annd reflection and radiation phases
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
    # getting index of frames which highest and lowest difference
    sorted = np.sort(delta_amplitude)
    sorted_index = np.argsort(delta_amplitude)
    if sorted_index[-1] > sorted_index[-2]:
        reflection_start_index = sorted_index[-2]
    else:
        reflection_start_index = sorted_index[-1]
    # reflection_start_index = np.argmax(delta_amplitude) + 5
    # Adding tolerance to the initial frame index to correct the reflection phase
    # reflection_start_index = np.argmax(delta_amplitude) + 5
    reflection_start_index = reflection_start_index - 5
    reflection_end_index = np.argmin(delta_amplitude) - 5
    print('reflection_start_index: ',reflection_start_index,'  reflection_end_index: ', reflection_end_index)
    # filtering relfection phase in experimental data 
    if reflection_start_index > reflection_end_index:
        filtered_data = data[:, :, 150:500 + 5]
    else:
        filtered_data = data[:, :, reflection_start_index:reflection_end_index]
    print('The size of filtered data:', filtered_data.shape)
    return reflection_start_index,reflection_end_index


def thermal_profile(experimental_data,mask,feature=3):
    """extracts thermal profile sequence of all features present in the segmentation mask

    Args:
        experimental_data (hd5): Thermographic data
        mask (numpy arry): segmentation mask
        feature (int, optional): Number of features. Defaults to 3.

    Returns:
        _type_:thermal sequence of all features 
    """
    # creating container to store temperature profile of respective features
    substrate=[]
    background=[]
    defects=[]
    thermal_band=[]   
    # performring masking operating to extract the respective feature data and taking a mean value for all time sequences
    if feature==3:
        for i in range(3,experimental_data.shape[2]):
            substrate.append(np.mean(experimental_data[:,:,i][mask==1]))
            background.append(np.mean(experimental_data[:,:,i][mask==0]))
            defects.append(np.mean(experimental_data[:,:,i][mask==2]))
            thermal_band.append(np.mean(experimental_data[:,:,i][mask[:,:]==3]))
        return substrate,background,defects,thermal_band
    else:
        for i in range(3,experimental_data.shape[2]):
            substrate.append(np.mean(experimental_data[:,:,i][mask==1]))
            background.append(np.mean(experimental_data[:,:,i][mask==0]))
            defects.append(np.mean(experimental_data[:,:,i][mask==2]))
        return substrate,background,defects,thermal_band    


def simulation(thermal_data, experiment, frames=1):
    """
    visualization of thermograms

    Args:
        thermal_data (_type_): Input thermal hadoop file
        experiment (_type_): name of the experiment 
        frames (int, optional): frames per second. Defaults to 1.
    """
    # extraction of experiment data from thermal hadoop file
    data = thermal_data[experiment]
    (m, n, t) = data.shape
    # creating timestep indices based on number of frames
    timesteps=range(0,t,frames)
    n=len(timesteps)
    # plotting the data using animation tool for effective visualization.
    fig = plt.figure()
    fig.set_size_inches(data.attrs['plot2DWidth'] / 2.54,
                            data.attrs['plot2DHeight'] / 2.54)
    def init():
        plt.clf()
        ax = sns.heatmap(data[:, :, timesteps[0]],cmap='RdYlBu_r')
        ax.set_xlabel('X pixcels (Width)')
        ax.set_ylabel('Y pixcels (Height)')
        ax.collections[0].colorbar.set_label('Temperature (\N{DEGREE SIGN}"C)')
        plt.title('Frame:' + str(0))
        plt.show(block=False)
    def animate(i):
        plt.clf()
        ax = sns.heatmap(data[:,:,timesteps[i+1]],cmap='RdYlBu_r')
        ax.collections[0].colorbar.set_label('Temperature (\N{DEGREE SIGN}C)')
        ax.set_xlabel('X pixcels (Width)')
        ax.set_ylabel('Y pixcels (Height)')
        plt.title('Frame:' + str(timesteps[i+1]))
        plt.show(block=False)
    # animation function which plots the heatmaps of thermograms.
    anima = animation.FuncAnimation(fig, animate,frames=n-1, init_func=init, interval=1000,repeat=False)
    plt.show()
    


