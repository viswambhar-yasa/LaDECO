# -*- coding: utf-8 -*-
## This file contain utilities required for smooth functioning of the software
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def thermal_profile(experimental_data,mask,feature=3):
    substrate=[]
    background=[]
    defects=[]
    thermal_band=[]   
    mask=mask[:255,:255]
    if feature==3:
        for i in range(3,experimental_data.shape[2]):
            substrate.append(np.mean(experimental_data[:255,:255,i][mask==1]))
            background.append(np.mean(experimental_data[:255,:255,i][mask==0]))
            defects.append(np.mean(experimental_data[:255,:255,i][mask==2]))
            thermal_band.append(np.mean(experimental_data[:255,:255,i][mask[:255,:255]==3]))
        return substrate,background,defects,thermal_band
    else:
        for i in range(3,experimental_data.shape[2]):
            substrate.append(np.mean(experimental_data[:255,:255,i][mask==1]))
            background.append(np.mean(experimental_data[:255,:255,i][mask==0]))
            defects.append(np.mean(experimental_data[:255,:255,i][mask==2]))
        return substrate,background,defects,thermal_band    
        
def thermal_profile_plot(experimental_data, mask, feature=3):
    """
    Continuous plotting of thermal profile of all feature present in the segmentation mask

    Args:
        experimental_data (3d array): thermographic data
        mask (2d array): segmentation mask
        feature (int, optional): Number of features. Defaults to 3.

    Returns:
        _type_: thermal profile of all features as list
    """
    # containers to hold thermal profile for each time step 
    substrate = []
    background = []
    defects = []
    thermal_band = []
    # sliceing the mask (to able to work on ascii generated and variotherm generated data)
    mask = mask[:255, :255]

    fig, ax = plt.subplots()
    def animate(i):
        if feature == 3:
            # applying mask to extract feature values and calculate the mean  
            substrate.append(np.mean(experimental_data[:255, :255, i][mask == 1]))
            background.append(np.mean(experimental_data[:255, :255, i][mask == 0]))
            defects.append(np.mean(experimental_data[:255, :255, i][mask == 2]))
            thermal_band.append(np.mean(experimental_data[:255, :255, i][mask[:255, :255] == 3]))
            ax.clear()
            ax.plot(substrate, color='tab:green', label='substrate')
            ax.plot(background, color='tab:blue', label='reference')
            ax.plot(defects, color='tab:red', label='damaged substrate')
            ax.plot(thermal_band, color='tab:orange', label='thermal_band')
            ax.set_xlabel('Time (Frames)')
            ax.set_ylabel('Temperature (K)')
            plt.grid()
            plt.legend()
        else:
            substrate.append(np.mean(experimental_data[:255, :255, i][mask == 1]))
            background.append(np.mean(experimental_data[:255, :255, i][mask == 0]))
            defects.append(np.mean(experimental_data[:255, :255, i][mask == 2]))
            ax.plot(substrate, color='tab:green', label='substrate')
            ax.plot(background, color='tab:blue', label='reference')
            ax.plot(defects, color='tab:red', label='damaged substrate')
            ax.set_xlabel('Time (Frames)')
            ax.set_ylabel('Temperature (K)')
            plt.grid()
            plt.legend()

    ani = animation.FuncAnimation(fig, animate, frames=(experimental_data.shape[2] - 1), interval=500, repeat=False)
    plt.show()
    return substrate, background, defects, thermal_band


def depth_constract_fucntion(substrate, background, defects, thermal_band, reflection_end_index, tol=8,
                             no_of_timesteps=200):
    """
    Constrast function based on equ 6.3.1 for depth evaluation
    Args:
        substrate (list): thermal profile of substrate
        background (list): thermal profile of reference
        defects (list): thermal profile of damaged substrate
        thermal_band (list): thermal profile of thermal band
        reflection_end_index (int): radiation start index
        tol (int, optional): To correc the frame at which radiation starts. Defaults to 8.
        no_of_timesteps (int, optional): The length of the radition phase. Defaults to 200.

    Returns:
        _type_: constrast function of all features
    """
    output = np.zeros(shape=(200, 3))
    constrast_sub = (np.array(substrate) - np.array(background)) / np.array(background)
    constrast_def = (np.array(defects) - np.array(background)) / np.array(background)
    constrast_thermal = (np.array(thermal_band) - np.array(background)) / np.array(background)
    output[:, 0] = np.squeeze(
        constrast_sub[reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps])
    output[:, 1] = np.squeeze(
        constrast_def[reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps])
    output[:, 2] = np.squeeze(
        constrast_thermal[reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps])
    output[np.isnan(output)] = 0
    output = np.expand_dims(output, axis=0)
    return output


def thermal_error_estimation(experimental_data):
    """
    To identify any anomalies in the thermographic data and identify the indices of those thermograms
    Args:
        experimental_data (3d array): thermographi data
    """
    # calculating the mean tempeature of each thermogram
    avg_temp = np.mean(experimental_data, axis=(0, 1))
    # calculating the step difference to identify any anomalies
    delta_temp = np.diff(avg_temp)
    # identify the reflection start and end phase
    reflection_index_st = np.argmax(delta_temp)
    reflection_index_end = np.argmin(delta_temp)
    # plotting the data to check for anomalies
    plt.figure(figsize=(10, 6))
    plt.plot(delta_temp, linewidth=2, linestyle='--', label='delta temperature profile')
    plt.xlabel('Time (frames)')
    plt.ylabel('step difference temperature (K)')
    plt.scatter(reflection_index_st, delta_temp[reflection_index_st], linewidth=3, color='tab:red',
                label='Reflection start point', marker='o')
    plt.scatter(reflection_index_end, delta_temp[reflection_index_end], linewidth=3, color='tab:green',
                label='Radiation start point', marker='o')
    plt.text(reflection_index_st, delta_temp[reflection_index_st] - 2, reflection_index_st)
    plt.text(reflection_index_end, delta_temp[reflection_index_end] + 2, reflection_index_end)
    plt.legend()
    plt.grid()
    plt.show()


def reflection_phase(experimental_data, index, disp=False):
    avg_temp = np.mean(experimental_data[:, :, index:-index], axis=(0, 1))
    delta_temp = np.diff(avg_temp)
    reflection_index_st = np.argmax(delta_temp)
    reflection_index_end = np.argmin(delta_temp)
    if disp:
        plt.figure(figsize=(10, 6))
        plt.plot(delta_temp, linewidth=2, linestyle='--', label='delta temperature profile')
        plt.xlabel('Time (frames)')
        plt.ylabel('step difference temperature (K)')
        plt.scatter(reflection_index_st, delta_temp[reflection_index_st], linewidth=3, color='tab:red',
                    label='Reflection start point', marker='o')
        plt.scatter(reflection_index_end, delta_temp[reflection_index_end], linewidth=3, color='tab:green',
                    label='Radiation start point', marker='o')
        plt.text(reflection_index_st, delta_temp[reflection_index_st] - 2, reflection_index_st)
        plt.text(reflection_index_end, delta_temp[reflection_index_end] + 2, reflection_index_end)
        plt.legend()
        plt.grid()
        plt.show()
    return reflection_index_st, reflection_index_end


def constrast_function(experimental_data, index, start_tol=10, constrast_val=271, x_crop_i=35, y_crop_i=25, x_crop_j=20,
                       y_crop_j=35, no_of_time_steps=200):
    """_summary_

    Args:
        experimental_data (3D array): thermographic data
        index (int): intital tolerance to slice anomalies
        start_tol (int, optional): index to adjust reflection phase frames. Defaults to 10.
        constrast_val (int, optional): to convert to a unique value. Defaults to 271.
        x_crop_i (int, optional): start index for width for cropping . Defaults to 35.
        y_crop_i (int, optional): start index for height for cropping . Defaults to 25.
        x_crop_j (int, optional): end index for width for cropping . Defaults to 20.
        y_crop_j (int, optional): end index for height for cropping . Defaults to 35.
        no_of_time_steps (int, optional): length of the radiation phase. Defaults to 200.

    Returns:
        _type_: constrast data for thickness evaluation
    """
    # cropping the input data
    input_data = experimental_data[x_crop_i:-y_crop_i, x_crop_j:-y_crop_j, :]
    # identifying reflection start and end index
    reflection_start_index, reflection_end_index = reflection_phase(input_data, index, disp=False)
    print("reflection_start_index: ", reflection_start_index, "reflection_end_index ", reflection_end_index)
    # calulating the mean temperature of initial phase
    based_temp = np.mean(input_data[:, :, start_tol:reflection_start_index - start_tol])
    # performing constrast operation
    id_data = (input_data[:, :,
               reflection_end_index + start_tol:reflection_end_index + start_tol + no_of_time_steps] - constrast_val) / constrast_val
    # reflection_length=(reflection_end_index-reflection_start_index)
    # normalized_data=std_output_data[:,:,reflection_length+10:reflection_length+10+no_of_time_steps]
    # converting 3D data to 2D sequence
    mean_inp = np.mean(id_data, axis=(0, 1))
    output_data = np.expand_dims(np.expand_dims(mean_inp, axis=0), axis=0)
    return output_data
