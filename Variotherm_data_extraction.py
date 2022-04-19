# -*- coding: utf-8 -*-
## Sample file to show the implementation of variotherm data extraction from .irb video
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)
import numpy as np
from thermograms.Data_extraction import VarioTherm

# file path
root_dir = r"utilites/datasets"
#experiment name 
experiment = r"2021-05-11 - Variantenvergleich - VarioTherm IR-Strahler - Winkel 45°"
# creating an object of variotherm api
vario_api = VarioTherm()
# extracting thermograms from .irb video
temperature_data_api = vario_api.image_sequence_extractor(root_dir, experiment, True)
# saving the extracted thermogram 
np.save(r'Documents/temp/variotherm_python_api.npy', temperature_data_api)
