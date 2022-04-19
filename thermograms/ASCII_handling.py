# -*- coding: utf-8 -*-
## This file contain data handling method to convert raw data into hadoop file format dataset 
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermography videos)

import os
import sys
import select

import numpy as np
import pandas as pd
import h5py
import gc

from thermograms.defaultconfig import config, cameraModel
from thermograms.Utilities import Utilities

class Data_Handling():
    '''
    Data Handling class to convert raw infrared thermal data at each timestep into 3D hadoop dataset dictionary of each training example
    '''
    
    def __init__(self, output_file_name, root_dir_path, default_config, cameraModel, evaluation_list_file_name=None, configuration_file=None) -> None:
        '''
        Initial method to create global variables

        Parameters
        ----------
        output_file_name : raw string
            Name of the hadoop file Ex: 'thermal_data'.
            
        root_dir_path : raw string
            path of directory which contains the data.
            
        default_config : Dictionary
            Contain default configuration values used to run program.
            
        cameraModel : Dictionary
            Contain default camera settings used to run program.
            
        evaluation_list_file_name : string, optional
            Contains the list of experiments which are converted. The default is None.
            
        configuration_file : string, optional
            Name of the experiment configuration present in each experiment. The default is None.

        '''
        self.output_file_name = output_file_name
        self.root_dir_path = root_dir_path
        self.default_config = default_config
        self.cameraModel = cameraModel
        if configuration_file is None: # If the file name need to be changed 
            self.configuration_file = 'evaluation-config.conf' # Default file name
        if evaluation_list_file_name is None:
            self.evaluation_list_file_name = 'evaluation-list.conf' # Default file name
        pass



    def create_directory(self, name, cd_disp=False)->None:
        '''
        Checks for a directory and creates it if not present

        Parameters
        ----------
        name : string
            Name of the directory.
            
        cd_disp : boolean, optional
            Parameter for printing the functionality in this method. The default is False.

        '''
        Utilities().create_directory(self.root_dir_path,name,cd_disp)
        pass


    def hadoop_file_generator(self, disp=False)->list:
        '''
        Check for the existence of the hadoop data file and create it along with listing the datasets within it.

        Parameters
        ----------
        disp : Boolean, optional
            For printitng the methods functionality. The default is False for no print.

        Returns
        -------
        experiment_in_file : list
            Contains the list of all experiments present in hadoop datafile.

        '''
        # Checks for a directory 'DATASET' to store datafile
        self.create_directory('Dataset', disp)
        output_file_path = os.path.join(
            self.root_dir_path, 'Dataset', self.output_file_name)
        # checks for the existances of the hadoop datafile
        if self.output_file_name not in os.listdir(os.path.join(
            self.root_dir_path, 'Dataset')):
            output_files = h5py.File(output_file_path, 'w')
            experiment_in_file = list(output_files.keys())
            if disp:
                print('The file doesnot exist, {} is created at {}'.format(
                    self.output_file_name, output_file_path),'\n')
            return experiment_in_file
        # if datafile exists list all the dataset with it
        else:
            output_files = h5py.File(output_file_path, 'r')
            experiment_in_file = list(output_files.keys())
            if disp:
                print('The file {}  exists at {}'.format(
                    self.output_file_name, os.path.join(
                        self.root_dir_path, 'Dataset')))
            print('LIST OF EXPERIMENTS IN FILE \n')
            for i in range(len(experiment_in_file)):
                print(i+1, ' : ', experiment_in_file[i])
            print('A total of {} experiments are loaded in file '.format(
                len(experiment_in_file)), '\n')
            return experiment_in_file


    def file_paths(self, experiment, camera, file_disp=False)->np.ndarray:
        '''
        Generates file paths based on the input data i.e directories/ sub-directories

        Parameters
        ----------
        experiment : string
            Name of the experiment.
            
        camera : string
            Type of camera ('OPTRIS' or 'VARIOTHERM').
            
        file_disp : Boolean, optional
            To print functionlities. The default is False.

        Returns
        -------
        files_path : numpy
            Contains the file path for all the file present in the directory.
            
        files_number : int
            Number of file in the directory.

        '''
        files_data_dir = os.path.join(self.root_dir_path, experiment, camera)
        # r=root, d=directories, f = files
        if camera == 'VarioTHERM':
            camera = 'VarioTherm'
        files_path = []
        for r, d, f in os.walk(files_data_dir):
            for file in f:
                if '.csv' in file and camera == 'Optris':
                    files_path.append(os.path.join(r, file))
                elif '.asc' in file and camera == 'VarioTherm':
                    files_path.append(os.path.join(r, file))
                else:
                    continue
        files_number = len(files_path)
        if file_disp:
            print('The number of files in {} are {}'
                  .format(experiment, str(files_number)))
        return files_path, files_number


    def read_configuration(self, experiment, read_disp=False):
        '''
        Creates a dictionary of experimental configuration present in experiment directory

        Parameters
        ----------
        experiment : strinf
            Name of the experiment.
            
        read_disp : Bool, optional
            To print the method functionality. The default is False.

        Returns
        -------
        config : Dictionary
            Contains the experimental configurations.

        '''
        config_file_path = os.path.join(
            self.root_dir_path, experiment, self.configuration_file)
        config = {}
        with open(config_file_path, 'r') as conf_data:
            for line in conf_data:
                line = line.lstrip()
                if line == '':
                    continue
                if line.lstrip()[0] == '#':
                    continue
                line = line.split('#')[0].rstrip()
                key = line.split('=')[0].rstrip()
                try:
                    value = eval(line.split('=')[1].lstrip().rstrip())
                except:
                    value = line.split('=')[1].lstrip().rstrip()
                config.update({key: value})
            config['name'] = experiment
        #if read_disp:
            #print(config)
        return config


    def read_raw_data_file(self, experiment_data, read_raw_disp=False):
        '''
        Reads CSV or ASCI data into pandas after performing the requires transformation and loaded them into hadoop file 

        Parameters
        ----------
        experiment_data : dictionary
            Consist of experiment configurations.
        read_raw_disp : Boolean, optional
            For printing method functionality. The default is False.

        Raises
        ------
        Exception
            If the camera type doesnot match with in the given list.

        '''
        if 'Optris' in experiment_data['Beschreibung']:
            camera = 'Optris'
        elif 'variotherm' in experiment_data['Beschreibung'].lower():
            camera = 'VarioTHERM'
        else:
            raise Exception('Camera status is not avaliable in Beschreibung')
        # extracting data for Camera configuration    
        image_config = self.cameraModel[camera]
        pixel_height = image_config["lines"]
        pixel_width = image_config["columns"]
        delimitor = image_config["delimiter"]
        decimal_sign = image_config["decimalSign"]
        temperature_scale = bool(image_config["changeTemperatureToCelsius"])
        camera = image_config["camera"]
        lines_to_skip = image_config['linesToSkip']
        experiment_name = experiment_data['name']
        files_path, files_number = self.file_paths(
            experiment=experiment_name, camera=image_config['dataPath'], file_disp=read_raw_disp)
        
        # Creating a numpy dataset to store data at time t
        temp_time_data = np.zeros(
            ((pixel_height-lines_to_skip), pixel_width, files_number), dtype='float16')
        
        # Loading data iterativly for all files with in the directory
        for i in range(files_number):
            if read_raw_disp and ((i+1) % 250) == 0:
                print('currently loading dataset :', i+1)
            # Reading data from CSV into pandas after performing transformation tasks    
            data = pd.read_csv(
                files_path[i], sep=delimitor, skiprows=int(lines_to_skip))
            if decimal_sign == ',':
                data = data.replace(',', '.', regex=True)
            data = data.dropna(axis=1)
            # converting pandas dataframe to numpy 
            temp_profile = data.to_numpy()

            if temperature_scale:
                temp_time_data[:, :, i] = temp_profile-273.15
            else:
                temp_time_data[:, :, i] = temp_profile
            del data  # dfs still in list
            gc.collect()
            ''' # For Plotting heatmap at time t
            if read_raw_disp:
                fig, ax = plt.subplots()
                ax = sns.heatmap(temp_time_data[:, :, i], cmap='RdYlBu_r')
                ax.set_xlabel(config['plot3DXLabel'])
                ax.set_ylabel(config['plot3DYLabel'])
                plt.title(experiment_data['Beschreibung'])
                plt.show(block=False)
                plt.pause(1)  # 3 seconds, I use 1 usually
                plt.close("all")
            '''
        # Transfering data for whole time length t into hadoop datafile    
        output_file_path = os.path.join(
            self.root_dir_path, 'Dataset', self.output_file_name)
        output_file = h5py.File(output_file_path, 'a')
        data = output_file.create_dataset(experiment_name, shape=(
            (pixel_height-lines_to_skip), pixel_width, files_number), dtype='float16', data=temp_time_data,
            compression="gzip", compression_opts=9)

        meta_data = {}
        meta_data.update(self.default_config)
        meta_data.update(image_config)
        meta_data.update(experiment_data)
        # Storing metadata for each experiment within the datafile
        for key, value in meta_data.items():
            try:
                data.attrs[key] = value
            except:
                data.attrs[key] = 'None'
            else:
                continue
        print('loaded {} files sucessfully \n'.format(files_number), '\n')
        pass
    def read_raw_data_file1(self, experiment_data, read_raw_disp=False):
        '''
        Reads CSV or ASCI data into pandas after performing the requires transformation and loaded them into hadoop file 

        Parameters
        ----------
        experiment_data : dictionary
            Consist of experiment configurations.
        read_raw_disp : Boolean, optional
            For printing method functionality. The default is False.

        Raises
        ------
        Exception
            If the camera type doesnot match with in the given list.

        '''
        if 'Optris' in experiment_data['Beschreibung']:
            camera = 'Optris'
        elif 'variotherm' in experiment_data['Beschreibung'].lower():
            camera = 'VarioTHERM'
        else:
            raise Exception('Camera status is not avaliable in Beschreibung')
        # extracting data for Camera configuration    
        image_config = self.cameraModel[camera]
        pixel_height = image_config["lines"]
        pixel_width = image_config["columns"]
        delimitor = image_config["delimiter"]
        decimal_sign = image_config["decimalSign"]
        temperature_scale = bool(image_config["changeTemperatureToCelsius"])
        camera = image_config["camera"]
        lines_to_skip = image_config['linesToSkip']
        experiment_name = experiment_data['name']
        files_path, files_number = self.file_paths(
            experiment=experiment_name, camera=image_config['dataPath'], file_disp=read_raw_disp)
        
        # Creating a numpy dataset to store data at time t
        temp_time_data = np.zeros(
            ((pixel_height-lines_to_skip), pixel_width, files_number), dtype='float16')
        
        # Loading data iterativly for all files with in the directory
        for i in range(files_number):
            if read_raw_disp and ((i+1) % 250) == 0:
                print('currently loading dataset :', i+1)
            # Reading data from CSV into pandas after performing transformation tasks    
            data = pd.read_csv(
                files_path[i], sep=delimitor, skiprows=int(lines_to_skip))
            if decimal_sign == ',':
                data = data.replace(',', '.', regex=True)
            data = data.dropna(axis=1)
            # converting pandas dataframe to numpy 
            temp_profile = data.to_numpy()

            if temperature_scale:
                temp_time_data[:, :, i] = temp_profile-273.15
            else:
                temp_time_data[:, :, i] = temp_profile
            del data  # dfs still in list
            gc.collect()
            ''' # For Plotting heatmap at time t
            if read_raw_disp:
                fig, ax = plt.subplots()
                ax = sns.heatmap(temp_time_data[:, :, i], cmap='RdYlBu_r')
                ax.set_xlabel(config['plot3DXLabel'])
                ax.set_ylabel(config['plot3DYLabel'])
                plt.title(experiment_data['Beschreibung'])
                plt.show(block=False)
                plt.pause(1)  # 3 seconds, I use 1 usually
                plt.close("all")
            '''
        # Transfering data for whole time length t into hadoop datafile    
        output_file_path = os.path.join(
            self.root_dir_path, 'Dataset', self.output_file_name)
        output_file = h5py.File(output_file_path, 'a')
        data = output_file.create_dataset(experiment_name, shape=(
            (pixel_height-lines_to_skip), pixel_width, files_number), dtype='float16', data=temp_time_data,
            compression="gzip", compression_opts=9)

        meta_data = {}
        meta_data.update(self.default_config)
        meta_data.update(image_config)
        meta_data.update(experiment_data)
        # Storing metadata for each experiment within the datafile
        for key, value in meta_data.items():
            try:
                data.attrs[key] = value
            except:
                data.attrs[key] = 'None'
            else:
                continue
        print('loaded {} files sucessfully \n'.format(files_number), '\n')
        pass

    def extraction_experiments_list(self):
        '''
        To extracts the name of experiments from the evaluation-list file and create a list

        Returns
        -------
        experiment_list : list
            name of the experiments to be evaluated.

        '''
        experiment_list = []
        exp_list_path = os.path.join(
            self.root_dir_path, self.evaluation_list_file_name)
        with open(exp_list_path, 'r') as file:
            for line in file:
                line = line.lstrip()
                if line == '':
                    continue

                if line.lstrip()[0] == '#':
                    continue
                else:
                    experiment_list.append(line.rstrip().lstrip())

        print("Number of experiments in evaluation list {}".format(
            len(experiment_list)), '\n')
        experiment_list = [y.replace('Â', '') for y in experiment_list]
        return experiment_list


    def load_data(self,experiment_list=None, disp=False):
        '''
        Load data into hadoop datafile based on the experiments obtained from evaluation list

        Parameters
        ----------
        experiment_list : list, optional
            list of experiment given in python not as in evaluation file. The default is None.
        disp : bool, optional
            To print method functionalities. The default is False.


        '''
        if experiment_list is None:
            experiment_list = self.extraction_experiments_list()
        # Created hadoop data file    
        experiments_in_file = self.hadoop_file_generator(disp)
        print('LOADING STARTED')
        # Iteratively running for all experiments in experiment_list 
        for i in range(len(experiment_list)):
            experiment = experiment_list[i]
            print(experiment)
            if experiment not in experiments_in_file: # checks if experiment is present in hadoop datafile or not
                experiment_data = self.read_configuration(experiment, disp)
                self.read_raw_data_file(experiment_data, disp)
            else:
                print('Experiment {} Data already present in the {} file'.format(
                    experiment_list[i], self.output_file_name))
                sys.stdout.write('Press Y to continue else N to load \n')
                sys.stdout.flush()
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 2)
                    load_condition = sys.stdin.readline().strip()
                except: # set of conditon to overwrite data or not 
                    ready = True
                    load_condition = input('Enter your option Y/N : ')
                if ready:
                    if load_condition == 'Y' or load_condition == 'y':
                        continue
                    elif load_condition == 'N' or load_condition == 'n':
                        experiment = experiment_list[i]
                        experiment_data = self.read_configuration(
                            experiment, disp)
                        output_file = h5py.File(self.output_file_name, 'a')
                        del output_file[experiment]
                        self.read_raw_data_file(experiment_data, disp)
                    else:
                        print('invalid option : Data is not overwritten')
                else:
                    continue
        
        print('LOADING COMPLETE')
        pass


if __name__ == '__main__':
    
    #config['PATH'] = 'D:\Thermal evaluation\Thermal_data'
    config['PATH'] = r'W:\ml_thermal_imaging\thermography_data\Daten Lackdetektion'
    main_dir = config['PATH']
    disp = False
    output_file_name = r'Variantenvergleich_data.hdf5'

    Variantenvergleich = Data_Handling(
        output_file_name, main_dir, config, cameraModel)
    Variantenvergleich.load_data(disp=True)
