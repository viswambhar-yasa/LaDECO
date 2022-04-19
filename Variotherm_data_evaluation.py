# -*- coding: utf-8 -*-
## Sample file to show the implementation of variotherm data loading of ascii data 
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

from thermograms.ASCII_evaluation import Data_Evaluation
from thermograms.Utilities import Utilities
# file path
root_path = r"utilites/datasets"
# ouput file name
data_file_name = r'Parameterstudie_data.hdf5'
thermal_object = Utilities()
thermal_data, experiment_list = thermal_object.open_file(root_path, data_file_name, True)
experiment=r'2021-11-24-Parameterstudie-1000W_Halogenstrahler-10s_Belichtungszeit'
# to perform data evaluation on thermographic data
variotherm = Data_Evaluation(thermal_data,experiment, root_path)

# for plotting thermograms
variotherm.simulation(frames=500)

# for plotting thermal profile
# indices of the features (x and y pixcel)
temperatureTimeCurves=["coating: 100/100","substrate: 50/40","thermal band: 80/80"]
variotherm.raw_temperaturevstime(box_size=3,evaluationPath=root_path,temperatureTimeCurves=temperatureTimeCurves,temp_disp=True)