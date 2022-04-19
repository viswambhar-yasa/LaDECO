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
from thermograms.Utilities import Utilities
from thermograms.Data_extraction import VarioTherm
import matplotlib.pyplot as plt


class Data_Handling():
    '''
    Data Handling class to convert raw infrared thermal data at each timestep into 3D hadoop dataset dictionary of each training example
    '''

    def __init__(self, output_file_name, root_dir_path, evaluation_list_file_name=None,
                 configuration_file=None) -> None:
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
        self.default_config = None
        self.phase_dict = None
        if configuration_file is None:  # If the file name need to be changed
            self.configuration_file = 'evaluation-config.conf'  # Default file name
        if evaluation_list_file_name is None:
            self.evaluation_list_file_name = 'evaluation-list.conf'  # Default file name
        pass

    def create_directory(self, name, cd_disp=False) -> None:
        '''
        Checks for a directory and creates it if not present

        Parameters
        ----------
        name : string
            Name of the directory.

        cd_disp : boolean, optional
            Parameter for printing the functionality in this method. The default is False.

        '''
        Utilities().create_directory(self.root_dir_path, name, cd_disp)
        pass

    def hadoop_file_generator(self, disp=False) -> list:
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
                    self.output_file_name, output_file_path), '\n')
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
                print(i + 1, ' : ', experiment_in_file[i])
            print('A total of {} experiments are loaded in file '.format(
                len(experiment_in_file)), '\n')
            return experiment_in_file

    def file_paths(self, experiment, camera, file_disp=False) -> np.ndarray:
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
        # if read_disp:
        #    print(config)
        self.default_config = config
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
        # extracting data for Camera configuration
        image_height = self.default_config["image_width"]
        image_width = self.default_config["image_height"]
        time_steps = self.default_config["time_steps"]
        temperature_scale = bool(self.default_config["changeTemperatureToCelsius"])
        camera = self.default_config["camera"]
        experiment_name = self.default_config['name']
        if temperature_scale:
            temp_time_data = experiment_data - 273.15
        else:
            temp_time_data = experiment_data
        output_file_path = os.path.join(
            self.root_dir_path, 'Dataset', self.output_file_name)
        output_file = h5py.File(output_file_path, 'a')
        data = output_file.create_dataset(experiment_name, shape=(image_height, image_width, time_steps),
                                          dtype='float16', data=temp_time_data,
                                          compression="gzip", compression_opts=9)
        meta_data = {}
        meta_data.update(self.default_config)
        meta_data.update(self.phase_dict)
        # Storing metadata for each experiment within the datafile
        for key, value in meta_data.items():
            try:
                data.attrs[key] = value
            except:
                data.attrs[key] = 'None'
            else:
                continue
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

    def phases_identification(self, experiment_data, experiment, phase_disp=True):
        tol_dtemp = self.default_config['temperaturDelta']
        IgnoreTimeAtStart = self.default_config['IgnoreTimeAtStart']
        if IgnoreTimeAtStart == 1:
            IgnoreTimeAtStart == 10
        temp_data = experiment_data[:, :, 1:]
        fourier_transformation = np.fft.fftn(temp_data, axes=(0, 1))
        amplitude = np.abs(fourier_transformation)
        amplitude_sequence = np.mean(amplitude, axis=(0, 1))
        delta_amplitude = np.diff(amplitude_sequence)


        #print(np.sort(delta_amplitude)[-5:])
        #print(np.argsort(delta_amplitude)[-5:])
        sorted = np.sort(delta_amplitude)
        sorted_index = np.argsort(delta_amplitude[:-5])
        if sorted_index[-1] > sorted_index[-2]:
            reflection_start_index = sorted_index[-2]
        else:
            reflection_start_index = sorted_index[-1]
        # reflection_start_index = np.argmax(delta_amplitude) + 5
        reflection_start_index=reflection_start_index-5
        reflection_end_index = np.argmin(delta_amplitude[:-5]) - 5
        print(reflection_start_index, reflection_end_index) 
        phase = {'reflection_phase_start_index': reflection_start_index,
                 'reflection_phase_end_index': reflection_end_index,
                 'radiation_phase_start_index': reflection_end_index + 5,
                 'radiation_phase_end_index': experiment_data.shape[2]}
        config_file_path = os.path.join(
            self.root_dir_path, experiment, self.configuration_file)
        self.phase_dict = phase
        with open(config_file_path, 'a') as f:
            f.write('reflection_phase_start_index =' + str(reflection_start_index) + '\n')
            f.write('reflection_phase_end_index =' + str(reflection_end_index) + '\n')
            f.write('radiation_phase_start_index =' + str(reflection_end_index + 5) + '\n')
            f.write('radiation_phase_end_index =' + str(experiment_data.shape[2]) + '\n')
        if phase_disp:
            print(phase)
        pass

    def load_data(self, experiment_list=None, disp=False):
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
            if experiment not in experiments_in_file:  # checks if experiment is present in hadoop datafile or not
                data_extraction = VarioTherm()
                experiment_data = data_extraction.image_sequence_extractor(self.root_dir_path, experiment, False)
                self.read_configuration(experiment, disp)
                self.phases_identification(experiment_data, experiment)
                self.read_raw_data_file(experiment_data, disp)
            else:
                print('Experiment {} Data already present in the {} file'.format(
                    experiment_list[i], self.output_file_name))
                sys.stdout.write('Press Y to continue else N to load \n')
                sys.stdout.flush()
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 2)
                    load_condition = sys.stdin.readline().strip()
                except:  # set of conditon to overwrite data or not
                    ready = True
                    load_condition = input('Enter your option Y/N : ')
                if ready:
                    if load_condition == 'Y' or load_condition == 'y':
                        continue
                    elif load_condition == 'N' or load_condition == 'n':
                        data_extraction = VarioTherm()
                        experiment = experiment_list[i]
                        experiment_data = data_extraction.image_sequence_extractor(self.root_dir_path, experiment,
                                                                                   False)
                        self.read_configuration(
                            experiment, disp)
                        output_file = h5py.File(self.output_file_name, 'a')
                        del output_file[experiment]
                        self.read_raw_data_file(experiment_data, disp)
                    else:
                        print('invalid option : Data is not overwritten')
                else:
                    continue
        print('\n LOADING COMPLETE')
        pass


if __name__ == '__main__':
    PATH = r'W:\ml_thermal_imaging\thermography_data\7.4 - Lackdetektion - Materialstudie\materials'
    # config['PATH'] = r'W:\ml_thermal_imaging\thermography_data\Daten Lackdetektion'
    # main_dir = config['PATH']
    disp = False
    output_file_name = r'material_thickness_1000W_5s.hdf5'
    Variantenvergleich = Data_Handling(output_file_name, PATH)
    Variantenvergleich.load_data(disp=False)
