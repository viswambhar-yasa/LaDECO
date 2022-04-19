# -*- coding: utf-8 -*-
## Sample file to show the implemtation of depth estimation module
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)


import numpy as np

print('Project MLaDECO')
print('Author: Viswambhar Yasa')
print('Software version: 0.1')
# importing functionality from different modules
from thermograms.Utilities import Utilities
from ml_training.dataset_generation.principal_componant_analysis import principal_componant_analysis
from utilites.thermal_profile import reflection_phase, thermal_profile_plot, depth_constract_fucntion
from utilites.segmentation_colormap_anno import segmentation_colormap_anno
from utilites.tolerance_maks_gen import tolerance_predicted_mask
from tensorflow.keras import models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# file path
root_path = r'utilites/datasets'
# file name
data_file_name = r'Variantenvergleich_data_python_api.hdf5'
# reading the hadoop file
thermal_object = Utilities()
thermal_data, experiment_list = thermal_object.open_file(root_path, data_file_name, True)
experiment_name = r'2021-05-25 - Variantenvergleich - VarioTherm Halogenlampe - Belichtungszeit 10s'
experimental_data = thermal_data[experiment_name]
# identifying the reflection phase
reflection_start_index, reflection_end_index = reflection_phase(experimental_data, index=5, disp=False)
# performing principal companant analysis to reduce 3D data to 2D data containing important characteristics
input_data = principal_componant_analysis(experimental_data)
print(input_data.shape)
# selection of segmentation model
model = 'U_net'
#loading machine learning model
if model == 'U_net':
    thermal_net = models.load_model(r'trained_models/UNet/u_net_64_Adam_5_best_agu.h5')
elif model == 'FC_net':
    thermal_net = models.load_model(r'trained_models/FCN/fc8_segmentation_adam_01_64.h5')
elif model == 'PSP_net':
    thermal_net = models.load_model(r'trained_models/PSP/PSP_adam_seg_16_01_model.h5')
else:
    thermal_net = models.load_model(r'trained_models/UNet/u_net_64_Adam_5_best_agu.h5')
# selection of standardizing method
scaling_type = 'normalization'
if scaling_type == 'normalization':
    standardizing = StandardScaler()
    std_output_data = standardizing.fit_transform(
        input_data.reshape(input_data.shape[0], -1)).reshape(input_data.shape)
else:
    normalizing = MinMaxScaler(feature_range=(0, 1))
    std_output_data = normalizing.fit_transform(
        input_data.reshape(input_data.shape[0], -1)).reshape(input_data.shape)

expanded_input_data = np.expand_dims(std_output_data, axis=0)
tolerance = 0.55
# predicting the segmentation maks
predicted_mask = thermal_net.predict(expanded_input_data)
final_mask = tolerance_predicted_mask(predicted_mask, tol=tolerance)
#plotting segmentation mask
segmentation_colormap_anno(final_mask, disp=True)
# calculating thermal profile based on the feature present in segmentation mask and plotting it
substrate, background, defects, thermal_band = thermal_profile_plot(experimental_data, final_mask, feature=3)
# calculating thr constrast function required to perform depth estimation
depth_input_data = depth_constract_fucntion(substrate, background, defects, thermal_band, reflection_end_index,tol=10)
# loading machine learning model
depth_network = models.load_model(r'trained_models/GRU/depth_estmation_Bi_GRU_adam.h5')
# predicting the depth of each features
depth_predicted = np.squeeze(depth_network.predict(depth_input_data))
print(depth_predicted)
# plotting the constrast function of each feature along with predicted depth.
plt.plot(np.squeeze(depth_input_data[:, :, 0]), color="tab:green", label='substrate: ' + str(depth_predicted[0]) + "mm")
plt.plot(np.squeeze(depth_input_data[:, :, 1]), color="tab:red",
         label='damaged substrate: ' + str(depth_predicted[1]) + "mm")
plt.plot(np.squeeze(depth_input_data[:, :, 1]), color="tab:orange",
         label='thermal band: ' + str(depth_predicted[2]) + "mm")
plt.xlabel("Time(frames)")
plt.ylabel("Contrast (Dimensionless)")
plt.title("Depth or elevation from coating given in legend")
plt.grid()
plt.legend()
plt.show()
