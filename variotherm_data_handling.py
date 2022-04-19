# -*- coding: utf-8 -*-
## Sample file to show the implementation of variotherm data loading of ascii data 
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

from thermograms.ASCII_handling import Data_Handling
from thermograms.Utilities import Utilities
# file path
root_path = r"utilites/datasets"
# ouput file name
data_file_name = r'test_sample.hdf5'
variotherm = Data_Handling(data_file_name, root_path)
# loads data based on the experiment present in the evalution list 
variotherm.load_data(disp=False)
# To open the loaded hadoop file and list the experiments
root_path = r"utilites/datasets/Dataset"
thermal_object = Utilities()
thermal_data, experiment_list = thermal_object.open_file(root_path, data_file_name, True)