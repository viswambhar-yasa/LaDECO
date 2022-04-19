# -*- coding: utf-8 -*-
## A sample code to show the implemtation of object segmentation module
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)
import numpy as np

print('Project MLaDECO')
print('Author: Viswambhar Yasa')
print('Software version: 0.1')

from thermograms.Utilities import Utilities
from ml_training.dataset_generation.fourier_transformation import fourier_transformation
from ml_training.dataset_generation.principal_componant_analysis import principal_componant_analysis
from utilites.segmentation_colormap_anno import segmentation_colormap_anno
from utilites.tolerance_maks_gen import tolerance_predicted_mask
from tensorflow.keras import models
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# file path
root_path = r'utilites/datasets'
# file name
data_file_name = r'Variantenvergleich_data_python_api.hdf5'
# opening hadoop file
thermal_object = Utilities()
# extrcting hadoop data and listing all experiments in it
thermal_data, experiment_list = thermal_object.open_file(root_path, data_file_name, True)
experiment_name = r'2021-05-25 - Variantenvergleich - VarioTherm Halogenlampe - Belichtungszeit 10s'
experimental_data = thermal_data[experiment_name]
# filtering dataset based on the identified reflection phase
input_data, reflection_st_index, reflection_end_index = fourier_transformation(experimental_data,
                                                                               scaling_type='normalization', index=1)
expanded_input_data = np.expand_dims(input_data, axis=0)
# selecting segmentation model
model = 'U_net'
features = 3
frames = 200
index = 0
print('Object segmentation model ', model)
print('Frames :', frames)
# creating a time step array based on frames as time step
timesteps = range(0, expanded_input_data.shape[3], frames)
# creating container to store segmentation mask
all_masks = np.zeros(shape=(input_data.shape[0], input_data.shape[1], len(timesteps)))
# loading ML model
if model == 'U_net':
    thermal_net = models.load_model(r'trained_models/UNet/u_net_64_Adam_5_best_agu.h5')
elif model == 'FC_net':
    thermal_net = models.load_model(r'trained_models/FCN/fc8_segmentation_adam_01_64.h5')
elif model == 'PSP_net':
    thermal_net = models.load_model(r'trained_models/PSP/PSP_adam_seg_16_01_model.h5')
else:
    thermal_net = models.load_model(r'trained_models/UNet/u_net_64_Adam_5_best_agu.h5')
# looped over timesteps
for i in timesteps:
    data = np.expand_dims(expanded_input_data[:, :, :, i], axis=-1)
    # predicting the segmentation mask
    predicted_mask = thermal_net.predict(data)
    # applying tolerance to penalize the probabilty
    final_mask = tolerance_predicted_mask(predicted_mask, tol=0.5)
    all_masks[:, :, index] = final_mask
    index += 1
    # plotting the segmentation mask for different features
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # fig.set_size_inches((experimental_data.attrs['plot2DWidth'] / 2.54),
    #                    (experimental_data.attrs['plot2DHeight'] / 2.54))
    sns.heatmap(np.squeeze(data[0, :, :, :]), cmap='RdYlBu_r', ax=ax1)
    ax1.set_xlabel('X pixcels (Width)')
    ax1.set_ylabel('Y pixcels (Height)')
    ax1.collections[0].colorbar.set_label('Temperature (Normalized)')
    ax1.axis('off')
    ax1.set_title("Input- thermogram")
    if features == 3:
        cmap = sns.mpl_palette("Set2", 4)
        sns.heatmap(data=final_mask, cmap=cmap, ax=ax2)
        plt.axis('off')
        legend_handles = [Patch(color=cmap[0], label='Coating'),  # red
                          Patch(color=cmap[1], label='Substrate'),
                          Patch(color=cmap[2], label='Damaged Substrate'),
                          Patch(color=cmap[3], label='Thermal band')]
        ax2.collections[0].colorbar.set_label(['0:Coating', '1:Substrate', '2:Damaged Substrate', '3:Thermal band'])
        ax2.set_title("Output- Segmentation")
        fig.tight_layout()
    else:
        cmap = sns.mpl_palette("Set2", 3)
        sns.heatmap(data=final_mask, cmap=cmap, ax=ax2)
        plt.axis('off')
        legend_handles = [Patch(color=cmap[0], label='Coating'),  # red
                          Patch(color=cmap[1], label='Substrate'),
                          Patch(color=cmap[2], label='Damaged Substrate')]
        ax2.collections[0].colorbar.set_label(['0:Coating', '  1:Substrate', '  2:Damaged Substrate'])
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(2.5)
    plt.close("all")
