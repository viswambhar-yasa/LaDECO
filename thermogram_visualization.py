# -*- coding: utf-8 -*-
## Sample file to show the implementation of Variotherm data simulation
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

from thermograms.Utilities import Utilities
from thermograms.Data_evaluation import simulation

print('Project MLaDECO')
print('Author: Viswambhar Yasa')
print('Software version: 0.1')

# file path 
root_path = r'utilites/datasets'
# file name
data_file_name = r'Parameterstudie_data.hdf5'
# to open hadoop data file
thermal_class = Utilities()
# extracts hadoop file and list all the experiments present in it 
thermal_data, experiment_list = thermal_class.open_file(root_path, data_file_name, True)
experiment_name = r'2021-12-02-Parameterstudie-1000W_Halogenstrahler-10s_Belichtungszeit'
# extrcting experiment data
experimental_data = thermal_data[experiment_name]
# simulation of thermograms of the extracted data
simulation(thermal_data, experiment_name, frames=500)