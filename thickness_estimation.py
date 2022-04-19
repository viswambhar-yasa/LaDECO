# -*- coding: utf-8 -*-
## Sample file to show the implementation of thickness estimation module
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)
import numpy as np

print('Project MLaDECO')
print('Author: Viswambhar Yasa')
print('Software version: 0.1')
from thermograms.Utilities import Utilities
from utilites.tolerance_maks_gen import tolerance_predicted_mask
from utilites.thermal_profile import constrast_function
from tensorflow.keras import models
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# file path 
root_path = r'utilites/datasets'
# file name
data_file_name = r'material_thickness_1000W.hdf5'
# to open the hadoop fle
thermal_object = Utilities()
# extracting hadoop file and listing experiment
thermal_data, experiment_list = thermal_object.open_file(root_path, data_file_name, True)
experiment_name = r'2021-12-07-Materialstudie-8.5-40µmS1013_40µmS6018_grün-1000W-10s'
experimental_data = thermal_data[experiment_name]
#length of the radiation phase
no_of_time_steps = 200
# calulating the constrast function 
input_data = constrast_function(experimental_data, index=8, start_tol=9, no_of_time_steps=200)
print("shape of input data", input_data.shape)
# assigning the thickness classes 
number_of_classes = 15
# assigning the thickness ranges
initial_thickness=0.001
final_thickness=0.1
# creating bins to classify thickness class
bins = np.linspace(initial_thickness, final_thickness, number_of_classes + 1)
# selecting a RNN model
model = 'Bi-GRU'
print('Thickness estimation model ', model)
# loading the model
if model == 'GRU':
    thickness_net = models.load_model(r'trained_models/GRU/thickness_estimation_GRU.h5')
elif model == 'LSTM':
    thickness_net = models.load_model(r'trained_models/LSTM/thickness_estimation_lstm.h5')
elif model == 'Bi-GRU':
    thickness_net = models.load_model(r'trained_models/GRU/thickness_estimation_bi_GRU.h5')
else:
    thickness_net = models.load_model(r'trained_models/LSTM/thickness_estimation_bi_lstm.h5')

#predicting the thickness class
thickness_class_predicted = np.argmax(thickness_net.predict(input_data), axis=1)
print("Thickness of the coating is", np.squeeze(bins[thickness_class_predicted]), " mm")
# plotting the constrast function along with the thickness class
plt.plot(np.squeeze(input_data), linewidth=2, color='tab:orange')
plt.ylabel("Constrast Function (No units)")
plt.xlabel("Radiation time frames")
plt.grid()
plt.title("Thickness of the coating is {} mm".format(np.squeeze(bins[thickness_class_predicted])))
plt.show()
