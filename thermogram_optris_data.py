# -*- coding: utf-8 -*-
## Sample file to show the implementation of optris data simulation
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

print('Project MLaDECO')
print('Author: Viswambhar Yasa')
print('Software version: 0.1')

from thermograms.Data_extraction import Optris

if __name__ == '__main__':
    # file path 
    PATH = r'utilites/datasets'
    # file name
    output_file_name = r'experiment_1.ravi'
    # creating an object
    Optris_extraction = Optris(PATH,output_file_name)
    # calling the simulation method 
    Optris_extraction.video_simulation(fps=30)
