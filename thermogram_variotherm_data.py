# -*- coding: utf-8 -*-
## Sample file to show the implementation of variotherm data extraction 
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)
print('Project MLaDECO')
print('Author: Viswambhar Yasa')
print('Software version: 0.1')

from thermograms.Data_processing import Data_Handling
# file path 
PATH = r'W:\ml_thermal_imaging\thermography_data\7.3 - Lackdetektion - Parameterstudie - IR-Quellen'
# ouput file name
output_file_name = r'Parameterstudie_data.hdf5'
# An object is created which takes the path and final output name
Variantenvergleich = Data_Handling(output_file_name, PATH)
# The file look into the evaluation list and load the experiment present in it 
Variantenvergleich.load_data(disp=False)
