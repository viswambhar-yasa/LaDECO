# -*- coding: utf-8 -*-
## This file contain data evaluation method to identify phases and features segmentation.
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)
import numpy as np
import matplotlib.pyplot as plt
import os
#import seaborn as sns
import pandas as pd
from thermograms.Utilities import Utilities,h5py


class Data_Evaluation:

    def __init__(self,data_set,experiment,root_dir):
        """

        Args:
            data_set (hd5): Thermograhic video dataset stored using Big Data
            experiment (str): name of the experiment
            root_dir (str): the path of the root directory where the data_set is present 
        """
        self.data_set=data_set
        self.experiment=experiment
        self.root_dir=root_dir
        self.experiment_data=data_set[experiment]
        self.points_positions=None
        self.phase_identification_dict=None
        self.intergal=None
        pass
    
    def simulation(self,frames=1)->None:
        """
        Visualization of the heat map extracted from thermographic video 

        Args:
            frames (int, optional): Frames per seconds. Defaults to 1.
        """
        # extracting time sequence length of the data  
        (m,n,t)=self.experiment_data.shape
        print('Simulation of '+self.experiment+'\n')
        # looping over time sequence with frames as steps length
        for i in range(0,t,frames):
            fig, ax = plt.subplots()
            fig.set_size_inches(self.experiment_data.attrs['plot2DWidth']/2.54,
                                self.experiment_data.attrs['plot2DHeight']/2.54)
            temp=self.experiment_data[:, :, i].astype(np.float64)
            ax = plt.imshow(temp, cmap='RdYlBu_r')
            cbar=plt.colorbar()
            cbar.set_label('Temperature ($^\circ$C)')
            plt.xlabel(self.experiment_data.attrs['plot3DXLabel'])
            plt.ylabel(self.experiment_data.attrs['plot3DYLabel'])
            plt.title(self.experiment_data.attrs['Versuchsbezeichnung']+' :'+str(i))
            plt.show(block=False)
            plt.pause(0.75)  
            plt.close("all")
        pass

    def raw_temperaturevstime(self,evaluationPath=None, file_save_path=None, box_size=1,temperatureTimeCurves=None, temp_disp=False, raw_save=False):
        """
        Extracting the temperature profile of all features

        Args:
            file_save_path (_type_, optional): path to save the extracted temperatute profile. Defaults to None.
            box_size (int, optional): size of the window for analysis. Defaults to 1.
            temp_disp (bool, optional): parameter to display temperature profile plot. Defaults to False.
            raw_save (bool, optional): parameter to save evaluated temperature profile. Defaults to False.

        Returns:
            _type_: temperature profile 
            _type_: position or index of the 
        """
        # container to store data
        # to store feature name
        name = []
        # to store it's index
        x = []
        y = []
        # Dictionary to store the temperature profile
        data_dict={}
        position_dict={}
        # extracting the metadata of file which contains the list of features and corresponding index
        if temperatureTimeCurves is None:
            temperatureTimeCurves=list(self.experiment_data.attrs['temperatureTimeCurves'])
        # running for all feaures in the list 
        for elements in temperatureTimeCurves:
            #splitting the str to extract name and index
            lists=elements.split(':')
            name.append(lists[0].rstrip().lstrip())
            x.append(lists[1].split('/')[0].rstrip().lstrip())
            y.append(lists[1].split('/')[1].rstrip().lstrip())
        # looping over all features obtained from the metadata
        if len(name)==0:
            print(name,x,y)
            raise Exception("temperatureTimeCurves is empty")

        for i in range(len(name)):
            if box_size==1:
                # extracting the temperature of the corresponding indices
                mean_temp = self.experiment_data[int(x[i]), int(y[i]), :]
            elif (box_size % 2) == 0:
                # operation to extract data for various box size (EVEN number)
                temp_time = self.experiment_data[int(x[i]):(
                    (int(x[i])+box_size-1)), int(y[i]):((int(y[i])+box_size-1)), :]
                mean_temp = np.mean(temp_time, axis=(0,1))
            else:
                box_length = int(box_size % 2)
                 # operation to extract data for various box size (ODD number)
                temp_time = self.experiment_data[(int(x[i])-box_length):(
                    int(x[i])+box_length), (int(y[i])-box_length):(int(y[i])+box_length), :]
                mean_temp = np.mean(temp_time, axis=(0,1))
            # loading extracted thermal profile of the features in the dictionary      
            data_dict[name[i]] = mean_temp
            # loading the position index of the features obtained from metadata of the file
            position_dict[name[i]] = [x[i],y[i]]
            #  Convert frame (frequency) to seconds
            time = np.arange(0, len(mean_temp))/self.experiment_data.attrs['frequency']
        # creating a table to store the above information 
        data_dict['time'] = time
        temp_tb_index=['x position','y position']
        position_table = pd.DataFrame(position_dict, index=temp_tb_index)
        raw_temperature_table = pd.DataFrame(data_dict)
        if evaluationPath is None:
            evaluationPath = self.experiment_data.attrs['evaluationPath']
        csv_name = 'raw_temperatureVstime.csv'
        # saving the csv at the path obtaine from evaluation path
        if file_save_path is None:
            csv_file_path = os.path.join(self.root_dir, self.experiment, 'csv-file')
        else:
            csv_file_path = os.path.join(file_save_path, self.experiment, 'csv-file')
        # create the folder if not present    
        Utilities().check_n_create_directory(csv_file_path)
        csv_path = os.path.join(csv_file_path, csv_name)
        if raw_save:
            # saves the data in csv format
            raw_temperature_table.to_csv(csv_path)

        if temp_disp:
            # plotting thermal profile of all the features
            raw_temperature_table.plot(kind='line',x='time')
            plt.grid(linestyle="--")
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature ($^\circ$C)')
            plt.title('Temperature vs Time ')
            plt.show()
            if file_save_path is None:
                plot_file_path = os.path.join(self.root_dir, self.experiment, evaluationPath)
            else:
                plot_file_path = os.path.join(file_save_path, self.experiment, evaluationPath)
            Utilities().check_n_create_directory(plot_file_path)
            plot_file_name='temperatureVstime.png'
            figure_path = os.path.join(plot_file_path, plot_file_name)
            if raw_save:
                # saving the plots at evaluation path 
                plt.savefig(
                    figure_path, dpi=self.experiment_data.attrs['plot3DDPI'], format=self.experiment_data.attrs['plot2DFileFormat'])
        
        self.points_positions=position_dict
        return raw_temperature_table, position_table

    def phases_identification(self, file_save_path=None, box_size=1, tol=0, raw_save=False, temp_disp=False):
        """
        phase identification from thermographic analysis 
        Args:
            file_save_path (str, optional): path to save data. Defaults to None.
            box_size (int, optional): size of the window (number of pixcels). Defaults to 1.
            tol (int, optional): initial tolerance for slicing data to remove errors. Defaults to 0.
            raw_save (bool, optional): Parameter to save data. Defaults to False.
            temp_disp (bool, optional): parameter to plot thermal profile. Defaults to False.

        Returns:
            _type_: phase index dictionary
        """
        # extracting metadata from file 
        tol_dtemp = self.experiment_data.attrs['temperaturDelta']
        # time step which have to be ignored for analysis
        IgnoreTimeAtStart = self.experiment_data.attrs['IgnoreTimeAtStart']
        if IgnoreTimeAtStart==0:
            IgnoreTimeAtStart=0.02
        # extracting thermal profile and position index of all the features    
        raw_temp, point_positions = self.raw_temperaturevstime(file_save_path, box_size, temp_disp,raw_save)
        temp_time_profile_tb = raw_temp.to_numpy()
        temp_profile = temp_time_profile_tb[:,:-1]
        point_name = point_positions.columns
        time_steps = temp_time_profile_tb[:, -1]
        index_of_ignore_time_start = np.where(time_steps == IgnoreTimeAtStart)
        delta_temp = np.diff(temp_profile, axis=0)
        filter_condition = (len(point_name))-tol
        # identificaton of the phases 
        reflection_phase = np.where(np.count_nonzero(
            delta_temp*(delta_temp >= tol_dtemp).astype(np.int), axis=1) >= filter_condition)
        reflection_phase_index=reflection_phase[0]
        start_index_of_reflection_phase = np.min(
            reflection_phase_index[reflection_phase_index > index_of_ignore_time_start[0][0]])
        if start_index_of_reflection_phase==0:
            start_index_of_reflection_phase=np.max(reflection_phase)
        radition_phase = np.where(np.count_nonzero(
            delta_temp*(delta_temp <= -tol_dtemp).astype(np.int), axis=1) >= filter_condition)[0]
        end_index_of_reflection_phase = np.min(radition_phase[radition_phase>index_of_ignore_time_start[0][0]])
        start_index_of_radition_phase = np.max(radition_phase)

        radition_phase_end = np.where(np.count_nonzero(
            delta_temp*((delta_temp >= -tol_dtemp/2) | (delta_temp <= -tol_dtemp/2) |(delta_temp == 0)).astype(np.int), axis=1) >= filter_condition)[0]
        end_index_of_radition_phase = np.max(radition_phase_end)

        # performing temperature offset by calculating mean and standard deviation of the initial phase temperature and removing it for reflection and radiation phase  
        temperature_offset = np.mean(
            temp_profile[index_of_ignore_time_start[0][0]:int(start_index_of_reflection_phase), :], axis=0)
        temperature_eplison = np.std(
            temp_profile[index_of_ignore_time_start[0][0]:int(start_index_of_reflection_phase), :], axis=0)
        max_delta_temp = np.max(
            delta_temp[index_of_ignore_time_start[0][0]:, :], axis=0)
        temp_profile_offset = temp_profile-temperature_offset
        temp_profile_offset[temp_profile_offset <= temperature_eplison]=0
        final_temp_profile = np.c_[temp_profile_offset, time_steps]
        temp_profile_offset_df = pd.DataFrame(
            final_temp_profile, columns=raw_temp.columns)


        # creating a dictionary to store the information about the phases 
        phase_dict={}
        phase_dict['name']=self.experiment
        phase_dict['frequency'] = self.experiment_data.attrs['frequency']
        phase_dict['time_step'] = 1/self.experiment_data.attrs['frequency']
        phase_dict['no_time_steps'] = len(time_steps)
        phase_dict['time_span'] = time_steps[len(time_steps)-1]
        phase_dict['reflection_phase_start_index'] = start_index_of_reflection_phase
        phase_dict['reflection_phase_end_index'] = end_index_of_reflection_phase
        phase_dict['radition_phase_start_index'] = start_index_of_radition_phase
        phase_dict['radition_phase_end_index'] = end_index_of_radition_phase
        phase_dict['reflection_phase_start_time'] = time_steps[start_index_of_reflection_phase]
        phase_dict['reflection_phase_end_time'] = time_steps[end_index_of_reflection_phase]
        phase_dict['radition_phase_start_time'] = time_steps[start_index_of_radition_phase]
        phase_dict['radition_phase_end_time'] = time_steps[end_index_of_radition_phase]
        phase_dict_df = pd.DataFrame.from_dict(phase_dict,orient='index')
        if raw_save:
            # saving the dictionary at the providen path 
            csv_name = 'phase_identification.csv'
            plot_file_name = 'offset_temperatureVstime.png'
            if file_save_path is None:
                file_path = os.path.join(self.root_dir, self.experiment, self.experiment_data.attrs['evaluationPath'])
            else:
                file_path = os.path.join(file_save_path, experiment, self.experiment_data.attrs['evaluationPath'])

            Utilities().check_n_create_directory(file_path)
            csv_path = os.path.join(file_path, csv_name)
            phase_dict_df.to_csv(csv_path)
            # plotting the thermal profile
            temp_profile_offset_df.plot(kind='line',x='time')
            plt.grid(linestyle="--")
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature ($^\circ$C)')
            plt.title('Temperature vs Time ')
            Utilities().check_n_create_directory(file_path)
            figure_path = os.path.join(file_path, plot_file_name)
            # saving the plot
            plt.savefig(
                figure_path, dpi=self.experiment_data.attrs['plot2DDPI'], format=self.experiment_data.attrs['plot2DFileFormat'])
        if temp_disp:
            print(phase_dict_df)
        cache = (temp_time_profile_tb, delta_temp,
                 temp_profile_offset_df, max_delta_temp)
        self.phase_identification_dict=phase_dict
        return phase_dict_df, cache

    def temperature_integral(self,file_save_path=None, box_size=1, tol=0, raw_save=False, temp_disp=False,plot_disp=False):
        '''
        programmed by Andreas Pestel

        Parameters
        ----------
        file_save_path : TYPE, optional
            DESCRIPTION. The default is None.
        box_size : TYPE, optional
            DESCRIPTION. The default is 1.
        tol : TYPE, optional
            DESCRIPTION. The default is 0.
        raw_save : TYPE, optional
            DESCRIPTION. The default is False.
        temp_disp : TYPE, optional
            DESCRIPTION. The default is False.
        plot_disp : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        energy_pixel : TYPE
            DESCRIPTION.

        '''
        phase_dict_df, cache= self.phases_identification(file_save_path, box_size, tol,raw_save,temp_disp)
        temp_time_profile_tb = cache[0]
        dict=phase_dict_df.to_dict()
        phase_dict=dict[0]
        if len(self.experiment_data.attrs['evaluationArea']) > 0:
            leftBorder = self.experiment_data.attrs['evaluationArea'][0]
            rightBorder = self.experiment_data.attrs['evaluationArea'][1]
            topBorder = self.experiment_data.attrs['evaluationArea'][2]
            bottomBorder = self.experiment_data.attrs['evaluationArea'][3]
        else:
            leftBorder = 0
            rightBorder = self.experiment_data.shape[0]
            topBorder = 0
            bottomBorder = self.experiment_data.shape[1]
        IgnoreTimeAtStart = self.experiment_data.attrs['IgnoreTimeAtStart']
        if IgnoreTimeAtStart == 0:
            IgnoreTimeAtStart = 0.02
        reflection_phase_start_index = phase_dict['reflection_phase_start_index']
        time_steps = temp_time_profile_tb[:, -1]
        index_of_ignore_time_start = np.where(time_steps == IgnoreTimeAtStart)
        evaluation_zone = self.experiment_data[leftBorder:rightBorder,
                               topBorder:bottomBorder, index_of_ignore_time_start[0][0]:reflection_phase_start_index]
        evaluation_zone_delta_temp=np.diff(evaluation_zone,axis=-1)
        temp_offset=np.mean(evaluation_zone,axis=-1)
        temp_epsilon=np.std(evaluation_zone,axis=-1)
        evaluation_zone_temp_offset = self.experiment_data-np.expand_dims(temp_offset,axis=-1)
        evaluation_zone_temp_offset = evaluation_zone_temp_offset * \
            (evaluation_zone_temp_offset >= np.expand_dims(
                temp_epsilon, axis=-1)).astype(int)
        max_temp=np.max(evaluation_zone_temp_offset)
        max_temp_time_index=np.where(max_temp==evaluation_zone_temp_offset)[-1][0]
        heating_phase = evaluation_zone_temp_offset[:, :,
                                                    index_of_ignore_time_start[0][0]:max_temp_time_index+1]

        median_temp_value=max_temp*0.5
        median_temp_condition = np.where(
            evaluation_zone_temp_offset >= median_temp_value)[-1]
        median_temp_condition=median_temp_condition*(median_temp_condition > max_temp_time_index).astype(int)
        median_delta_temp_condition = np.where(evaluation_zone_delta_temp >= self.experiment_data.attrs['temperaturDelta'])[-1]
        median_delta_temp_condition = median_delta_temp_condition * \
            (median_delta_temp_condition > max_temp_time_index).astype(int)
        if median_temp_condition.size ==0 or median_delta_temp_condition.size==0:
            cooling_phases_end_index=0
        else:
            cooling_phases_end_index = np.max(
            median_temp_condition[-1],median_delta_temp_condition[-1])

        cooling_phase = evaluation_zone_temp_offset[:, :,
                                                    max_temp_time_index+1:cooling_phases_end_index]

        restOfCurve = evaluation_zone_temp_offset[:, :,cooling_phases_end_index+1:]

        try:
            heatingPhase = np.sum(heating_phase,axis=-1) / self.experiment_data.attrs['frequency']
        except:
            heatingPhase = 0.0

        try:
            riseOfHeatingPhase = (
                heating_phase[:, :,-1] - heating_phase[:, :, 0]) / heating_phase.shape[2] / self.experiment_data.attrs['frequency']
        except:
            riseOfHeatingPhase = 0.0

        try:
            coolingPhase = np.sum(cooling_phase, axis=-1) / self.experiment_data.attrs['frequency']
        except:
            coolingPhase = 0.0

        try:
            riseOfcoolingPhase = (
                cooling_phase[:, :, -1] - cooling_phase[:, :, 0]) / cooling_phase.shape[2] / self.experiment_data.attrs['frequency']
        except:
            riseOfcoolingPhase = 0.0

        try:
            riseOfRestCurve = (np.max(restOfCurve,axis=-1) - np.min(restOfCurve,axis=-1)) / \
                                       (restOfCurve.shape[2]) / self.experiment_data.attrs['frequency']
        except:
            riseOfRestCurve = 0.0

                # Numersiches Integral der verbliebenden Kurve
        try:
            restOfCurve = np.sum(restOfCurve,axis=-1) / self.experiment_data.attrs['frequency']
        except:
            restOfCurve = 0.0

                # Numerisches Integral über gesamten Verlauf bilden
        energy_pixel = np.sum(evaluation_zone_temp_offset,axis=-1) / self.experiment_data.attrs['frequency']
        if plot_disp:
            if file_save_path is None:
                file_path = os.path.join(self.root_dir, self.experiment, self.experiment_data.attrs['evaluationPath'])
            else:
                file_path = os.path.join(file_save_path, self.experiment, self.experiment_data.attrs['evaluationPath'])
            key='Thermalprofile Integral'
            #plot3D(data,pixel,plot_file_path,key)
            fig, ax = plt.subplots()
            fig.set_size_inches(self.experiment_data.attrs['plot2DWidth']/2.54,
                                self.experiment_data.attrs['plot2DHeight']/2.54)
            ax = plt.imshow(energy_pixel, cmap='RdYlBu_r')
            cbar=plt.colorbar()
            cbar.set_label(self.experiment_data.attrs['plot3DZLabelIntegral'])
            plt.xlabel(self.experiment_data.attrs['plot3DXLabel'])
            plt.ylabel(self.experiment_data.attrs['plot3DYLabel'])
            plt.title(self.experiment_data.attrs['Versuchsbezeichnung']+' '+key)
            plot_name='temperature_profile_integral.png'
            figure_path=os.path.join(file_path,plot_name)
            plt.savefig(
                figure_path, dpi=self.experiment_data.attrs['plot2DDPI'], format=self.experiment_data.attrs['plot2DFileFormat'])
        self.integral=energy_pixel
        return energy_pixel


if __name__ == '__main__':
    
    # sample code to check the training data
    root_path = r'utilites/datasets'
    data_file_name = r'Variantenvergleich_data_python_api.hdf5'
    a = Utilities()
    thermal_data,experiment_list=a.open_file(root_path, data_file_name,True)
    #experiment = '2021-05-11 - Variantenvergleich - VarioTherm Halogenlampe - Winkel 45°'
    for i in experiment_list.keys():
        experiment=experiment_list[i]
        print('\n EVALUATION ')
        print(str(i)+' :'+experiment+'\n')
        thermal_evaluation=Data_Evaluation(thermal_data,experiment,root_path)
        thermal_evaluation.raw_temperaturevstime()
        phase_index,x=thermal_evaluation.phases_identification()
        eng_integral=thermal_evaluation.temperature_integral(raw_save=True,temp_disp=True,plot_disp=True)
        thermal_evaluation.simulation(frames=250)


