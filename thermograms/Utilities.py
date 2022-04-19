# -*- coding: utf-8 -*-
## This file contain utilities required for smooth functioning of the software
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

import os
import h5py


class Utilities:
    def __init__(self):
        pass

    def create_directory(self, root_directory, name, cd_disp=False):
        '''
        Checks for a directory and creates it if not present

        Parameters
        ----------
        name : string
            Name of the directory.

        cd_disp : boolean, optional
            Parameter for printing the functionality in this method. The default is False.

        '''
        directory_path = os.path.join(root_directory, name)
        check_existance = os.path.exists(directory_path)
        if cd_disp:
            print('Directory {} is already exists, the path is {}'.format(
                name, directory_path), '\n')
        if not check_existance:
            os.makedirs(directory_path)
            if cd_disp:
                print('Directory {} is created, the path is {}'.format(
                    name, directory_path), '\n')
        pass

    def open_file(self, root_directory, file_name, of_disp=False):
        """

        Args:
            root_directory (str): path of the file
            file_name (str): file name
            of_disp (bool, optional): parameter to print the file in the hadoop file. Defaults to False.

        Returns:
            _type_: hadoop file and list of data in the hadoop file
        """
        data_file_path = os.path.join(root_directory, file_name)
        data_file = h5py.File(data_file_path, 'r')
        data_list = {}

        print('Experiments in the file \n')
        data_in_file = list(data_file.keys())
        for i in range(len(data_in_file)):
            if of_disp:
                print(i + 1, ' : ', data_in_file[i])
            data_list[i] = data_in_file[i]
        print('\n')
        print('A total of {} experiments are loaded in file '.format(
            len(data_list)), '\n')
        return data_file, data_list


    def check_n_create_directory(self, file_path, cnc_disp=False):
        """

        Args:
            file_path (str): path of the file
            cnc_disp (bool, optional): parameter for printing. Defaults to False.
        """
        if os.path.exists(file_path):
            if cnc_disp:
                print('directory exists')
        else:
            os.makedirs(file_path)
            if cnc_disp:
                print('directory does not exists, thus creating it {}'.format(file_path))
