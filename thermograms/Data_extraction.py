# -*- coding: utf-8 -*-
## This file contain data extraction module extracts thermograms from Variotherm sensor camera and optris sensor camera
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import cv2 as cv

from thermograms.Utilities import Utilities

class VarioTherm():
    """Extraction of thermograms throught thermal sensor processing
    """
    def __init__(self) -> None:
        """ Intitial parameters to perform thermal sensor processing
        """
        self.thermo_video_data = None
        self.file_extension = None
        self.file_type = None
        self.file_type_extension = None
        self.sequence_step_data = {}
        self.max_length = None
        self.m_index = 0
        self.sequence_offset = 0
        self.sequence_count = 0
        self.sequence_count_max = 0
        self.no_time_seq = 0
        self.image_data = None
        self.image_length = 0
        self.image_index = 0
        self.image_width = 0
        self.image_height = 0
        pass

    def read_byte(self, index=0, length=4, type='int', byteorder='little'):
        """
        Read byte and conert it into the required format
        Args:
            index (int, optional): stary index of the byte . Defaults to 0.
            length (int, optional): length of the byte. Defaults to 4.
            type (str, optional): type to which the byte needs to be converted. Defaults to 'int'.
            byteorder (str, optional): type of byte order. Defaults to 'little'.

        Returns:
            _type_: value after conversion
        """
        # checking if the index of the byte exceeds the lenght of the file
        if index + length > self.max_length:
            # extracting the byte data from the main file
            temp = self.thermo_video_data[index:self.max_length - length]
            # updating the global index to max length
            self.m_index = self.max_length
        else:
            # extracting the byte data from the main file
            temp = self.thermo_video_data[index:index + length]
            # updating the global index
            self.m_index = index + length
        # converting the extracted byte information to required data format 
        if type == 'int':
            output = int.from_bytes(temp, byteorder, signed=True)
            return output
        elif type == 'float':
            output = struct.unpack('f', temp)
            return output[0]
        else:
            return temp

    def image_read_byte(self, index=0, length=2, type='int', byteorder='little'):
        """
        Reads the image byte and convert it into required data format

        Args:
            index (int, optional): _description_. Defaults to 0.
            length (int, optional): _description_. Defaults to 2.
            type (str, optional): _description_. Defaults to 'int'.
            byteorder (str, optional): _description_. Defaults to 'little'.

        Returns:
            _type_: convered data 
        """
        # checking if the index of the byte exceeds the lenght of the file
        if index + length > self.image_length:
            # extracting the byte data from the main file
            temp = self.image_data[index:self.image_length - length]
            # updating the image index to max length
            self.image_index = self.image_length
        else:
            # extracting the byte data from the main file
            temp = self.image_data[index:index + length]
            # updating the image index 
            self.image_index = index + length
        # converting the extracted byte information to required data format 
        if type == 'int':
            output = int.from_bytes(temp, byteorder, signed=True)
            return output
        elif type == 'float':
            output = struct.unpack('f', temp)
            return output[0]
        elif type == 'double':
            output = struct.unpack('d', temp)
            return output[0]
        else:
            return temp

    def set_index(self, index):
        """updates the global index 

        Args:
            index (int): index position which needs to be updated to global index
        """
        # checking if the index is in between the range of binary file 
        if self.max_length > index > 0:
            self.m_index = index
        # if the index is greater then max_length
        elif index >= self.max_length:
            self.m_index = self.max_length
        else:
            self.m_index = 0

    def sequence_block_data(self, disp=False):
        """
        Extraction of sequence information of thermogram which are required for 
        identifying the length and position of the thermogram in the binary file

        Args:
            disp (bool, optional): parameter to print dataa. Defaults to False.
        """
        # looping over the max sequence counter
        for i in range(self.sequence_count_max):
            # extraction of respective data 
            data = {'step_type': self.read_byte(self.m_index), 'description1': self.read_byte(self.m_index),
                    'frame_index': self.read_byte(self.m_index), 'step_offset': self.read_byte(self.m_index),
                    'step_size': self.read_byte(self.m_index), 'header_size': 0x6C0}
            if data['header_size'] > data['step_size']:
                data['header_size'] = data['step_size']
            data['header_offset'] = 0
            data['image_offset'] = data['header_size']
            data['image_size'] = data['step_size'] - data['image_offset']
            data['description2'] = self.read_byte(self.m_index)
            data['description3'] = self.read_byte(self.m_index)
            data['description4'] = self.read_byte(self.m_index)
            if data['step_type'] == 1:
                # creating a numpy array to store the respective thermogram information
                self.sequence_step_data[self.no_time_seq + 1] = data
                self.no_time_seq += 1
                if data['frame_index'] % 50 == 0 and disp:
                    print(self.sequence_step_data[self.no_time_seq])
        # To avoid the last two thermograms 
        self.no_time_seq = self.no_time_seq - 2
        pass

    def video_info_extraction(self, info_index=1084):
        """
        Extract thermal sensor parameter present in the binary file

        Args:
            info_index (int, optional): start index for video informartion. Defaults to 1084.

        Returns:
            _type_: dictionary contain thermal sensor information 
        """
        # Dictionary to store and update sensor information 
        video_info = {}
        self.image_index = info_index + 92
        device_min_range = self.image_read_byte(self.image_index, length=4, type='float')
        device_max_range = self.image_read_byte(self.image_index, length=4, type='float')
        video_info['device_min_range'] = str(device_min_range)
        video_info['device_max_range'] = str(device_max_range)
        self.image_index += 42
        device = self.image_read_byte(self.image_index, length=10, type='str')
        video_info['device'] = str(device)
        self.image_index += 34
        device_series_number = self.image_read_byte(self.image_index, length=6, type='str')
        video_info['device_series_number'] = str(device_series_number)
        self.image_index += 10
        sensor = self.image_read_byte(self.image_index, length=12, type='str')
        video_info['sensor'] = str(sensor)
        self.image_index += 18
        sensor_calibration = self.image_read_byte(self.image_index, length=32, type='str')
        video_info['sensor_calibration'] = str(sensor_calibration)
        self.image_index += 276
        video_timestamp = self.image_read_byte(self.image_index, length=8, type='double')
        video_timestamp_extension = self.image_read_byte(self.image_index, length=4, type='int')
        self.image_index += 2
        sensor_name = self.image_read_byte(self.image_index, length=10, type='str')
        video_info['camera'] = str(sensor_name)
        self.image_index += 45
        video_format = self.image_read_byte(self.image_index, length=16, type='str')
        video_info['video_format'] = str(video_format)
        return video_info

    def data_file_reading(self, root_dir, experiment, read_mode='rb'):
        """
        Reads the .irb file format and convert it into binary file format
        after which the data is extracted
        Args:
            root_dir (str): path of the file
            experiment (str): name of the file
            read_mode (str, optional): convert to the required file format. Defaults to 'rb'.(raw binary file format)
        """
        # file path of the video
        video_file_path = os.path.join(
            root_dir, experiment, experiment + '.irb')
        # reading the video and converting it into binary file format     
        with open(video_file_path, read_mode) as file:
            self.thermo_video_data = file.read()
        # print('The length of the sequence',len(self.thermo_video_data))
        # extracting initial video parameter 
        self.max_length = len(self.thermo_video_data)
        self.file_extension = self.read_byte(self.m_index, length=5, type='str')
        if self.file_extension != b"\xFFIRB\x00":
            print('File extension is not irb')
        self.file_type = self.read_byte(self.m_index, length=8, type='str')
        self.file_type_extension = self.read_byte(self.m_index, length=8, type='str')
        self.initial_flag = self.read_byte(self.m_index)
        self.sequence_offset = self.read_byte(self.m_index)
        self.sequence_count = self.read_byte(self.m_index)
        self.sequence_count_max = self.sequence_count + self.initial_flag
        self.set_index(self.sequence_offset)
        # extrating the thermogram sequence data based on the above initial information
        self.sequence_block_data()
        print('Number of time steps:', self.no_time_seq)
        pass

    def image_extraction(self, data_dic, root_dir, experiment, disp=False):
        """
        Extraction of thermogram data based on the data obtained in sequence_block_data

        Args:
            data_dic (numpy array): image sequence data obtained in sequence_block_data
            root_dir (str): path to save the image information
            experiment (str): name of the experiment 
            disp (bool, optional): parameter to print the image information . Defaults to False.

        Returns:
            _type_: thermogram
        """
        # extracting image sequence information like start index of the thermogram and length
        index = data_dic['step_offset']
        size = data_dic['step_size']
        frame_index = data_dic['frame_index']
        # print(type(index),size)
        # creating a dictionary to store thermogram information like width ,height etc
        image_info = {}
        self.image_length = size
        self.image_data = self.read_byte(index, size, 'str')
        # print(self.image_data)
        image_info['image_size'] = self.image_length
        self.image_index = 0
        bytes_per_pixel = self.image_read_byte(self.image_index, length=1, byteorder='big')
        compressed = self.image_read_byte(self.image_index, length=2, byteorder='big')
        image_info['bytes_per_pixel'] = str(bytes_per_pixel)
        image_info['compressed'] = str(compressed)
        self.image_index += 2
        self.image_width = self.image_read_byte(
            self.image_index, length=2, type='int', byteorder='big')
        self.image_height = self.image_read_byte(
            self.image_index, length=2, type='int', byteorder='big')
        self.image_index += 4
        image_info['image_width'] = str(self.image_width)
        image_info['image_height'] = str(self.image_height)
        image_info['time_steps'] = str(self.no_time_seq)
        width_check = self.image_read_byte(
            self.image_index, length=2, type='int', byteorder='big')
        # if width_check == image_width-1:
        #    raise Exception('width donot match')
        self.image_index += 2
        height_check = self.image_read_byte(
            self.image_index, length=2, type='int', byteorder='big')
        # if height_check == image_height-1:
        #    raise Exception('height donot match')
        self.image_index += 5
        emissivity = self.image_read_byte(self.image_index, length=4,
                                          type='float', byteorder='big')
        image_info['emissivity'] = str(emissivity)
        distance = self.image_read_byte(self.image_index, length=4,
                                        type='float', byteorder='big')
        image_info['distance'] = str(distance)
        environment_temp = self.image_read_byte(self.image_index, length=4,
                                                type='float', byteorder='big')
        self.image_index += 4
        path_temperature = self.image_read_byte(self.image_index, length=4,
                                                type='float', byteorder='big')
        image_info['path_temperature'] = str(path_temperature)
        self.image_index += 4
        center_wavelength = self.image_read_byte(self.image_index, length=4,
                                                 type='float', byteorder='big')
        image_info['center_wavelength'] = str(center_wavelength)
        self.image_index = 60 

        interpolation_temp = []
        # the temperatures in the thermogram are stores in sequential format 
        ## where two adjust value have to be interpolated to obtain the true temperature
        for i in range(256):
            # converting the byte data to float and appending to a list
            interpolation_temp.append(
                self.image_read_byte(self.image_index, length=4, type='float', byteorder='little'))
        # extraction of thermal sensor data
        video_info = self.video_info_extraction()
        # interpolating the temperature data to get true temperature.
        temperature_data = self.temperature_interpolation(data_dic['image_offset'], interpolation_temp)
        if frame_index == 1:
            # exporting the video information into a config file
            csv_name = 'evaluation-config.conf'
            # Utilities().check_n_create_directory(file_path)
            txt_file_path = os.path.join(
                root_dir, experiment)
            csv_path = os.path.join(txt_file_path, csv_name)
            with open(csv_path, 'w') as f:
                f.write("# -*- coding: utf-8 -*- \n")
                f.write('Versuchsbezeichnung =  \n')
                f.write('Beschreibung = \n')
                f.write('# Allgemein \n')
                for key in video_info.keys():
                    f.write(str(key) + "=" + str(video_info[key]) + "\n")
                for key in image_info.keys():
                    f.write(str(key) + "=" + str(image_info[key]) + "\n")
                f.write('changeTemperatureToCelsius = False \n')
                f.write('frequency=50 \n')
                f.write('plot3DElevation = 65				\n')
                f.write('plot3DAzimuth = None				\n')
                f.write('plot3DXLabel = Width [Pixel]		\n')
                f.write('plot3DYLabel = Height [Pixel]			\n')
                f.write('plot3DZLabelIntegral = \n')
                f.write('plot3DZLabelRise = m [K/s]			\n')
                f.write('plot3DWidth = 16.0					\n')
                f.write('plot3DHeight = 12.0					\n')
                f.write('plot3DDPI = 300						\n')
                f.write('plot3DFileFormat = png				\n')
                f.write('plot2DWidth = 16.0					\n')
                f.write('plot2DHeight = 12.0					\n')
                f.write('plot2DDPI = 300						\n')
                f.write('plot2DFileFormat = png				\n')
                f.write('evaluationArea = []                \n')
                f.write('temperatureTimeCurves =[]           \n')
                f.write('IgnoreTimeAtStart = 0 \n')
                f.write('temperaturDelta =1 \n')
            # video_info_df = pd.DataFrame.from_dict(video_info,orient='index')
            # image_info_df = pd.DataFrame.from_dict(image_info, orient='index')
            # evaluation_configuration = pd.concat([video_info_df, image_info_df])
            # evaluation_configuration.to_csv(csv_path)
        if disp and (frame_index % 10 == 0):
            # plots the heat map of thermogram
            plt.imshow(temperature_data.reshape(
                (self.image_width, self.image_height)).astype(np.float64), cmap='RdYlBu_r')
            plt.title('Temperature profile ' + str(frame_index) + ' time step')
            plt.xlabel('Height (pixcels)')
            plt.ylabel('width (pixcels)')
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.75)
            plt.close("all")
        return temperature_data

    def temperature_interpolation(self, index, interpolation_temp):
        """
        Interpolation function to map value of thermogram to obtain true temperature

        Args:
            index (_type_): start index for interpolation values
            interpolation_temp (_type_): list of temperatures which have to be interploated

        Returns:
            _type_: True thermograms
        """
        no_pixcels = self.image_height * self.image_width
        temperature_data = []
        f = 0
        self.image_index = index
        # runs for the number of pixcels
        for i in range(no_pixcels):
            # reads the pixcel positons(x,y)
            pixcel_1 = self.image_read_byte(self.image_index, length=1, type='int', byteorder='big')
            pixcel_2 = self.image_read_byte(self.image_index, length=1, type='int', byteorder='big')
            # interpolation function obtained from general data processing of thermal sensor data 
            f = pixcel_1 * (1.0 / 256.0)
            pixcel_temperature = interpolation_temp[pixcel_2 +
                                                    1] * f + interpolation_temp[pixcel_2] * (1.0 - f)
            # if the true temperature is less 0 K, then min range of the sensor is assigned 
            if pixcel_temperature < 0:
                pixcel_temperature = 255.0
            temperature_data.append(pixcel_temperature)
        return np.array(temperature_data)

    def image_sequence_extractor(self, root_dir, experiment, disp=False):
        """
        Combine all the above methods to extract the thermogram and store it in numpy array

        Args:
            root_dir (_type_): path of the .irb file 
            experiment (_type_): _description_
            disp (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # reads the data and convert into binary format and extracts image sequence information like start,lenght indices of thermogram
        self.data_file_reading(root_dir, experiment)
        # creating a numpy array to store the thermograms
        image_sequence = np.zeros(shape=(256, 256, len(self.sequence_step_data)-2))
        print('\nExtracting temperature profile sequence')
        print('Progress: [', end='', flush=True)
        # running for all sequences
        for i in range(1, len(self.sequence_step_data)-2):
            # extracting image information for each block
            data_dic = self.sequence_step_data[i]
            # extracting thermogram of each sequence 
            step_imag_temp = self.image_extraction(data_dic, root_dir, experiment, disp)
            # reshaping the extracted thermogram based on the extracted image width and height.
            image_sequence[:, :, i-1] = step_imag_temp.reshape((self.image_width, self.image_height))
            if i % 10 == 0:
                print('■', end='', flush=True)
        print('] loaded ' + str(self.no_time_seq) + ' time steps', end='', flush=True)
        return image_sequence


class Optris():
    """
    Processing of thermal sensor data
    """
    def __init__(self, root_directory, video_file_name) -> None:
        """
        initial parameterss

        Args:
            root_directory (str): path of the .ravi file
            video_file_name (str): name of the .ravi file
        """
        self.Root_directory = root_directory
        self.Ravi_file_name = video_file_name
        pass

    def video_simulation(self, fps=30, vs_disp=False):
        """
        Simulation of optris thermal video file
        Args:
            fps (int, optional): Frames per second. Defaults to 30.
            vs_disp (bool, optional): Parameter to perform simulation. Defaults to False.

        Raises:
            Exception: file is not in .ravi file format
        """
        # path of the ravi file
        video_file_path = os.path.join(self.Root_directory, self.Ravi_file_name)
        # using open CV .avi module to open data
        ravi_video_data = cv.VideoCapture(video_file_path)
        # changing the format of the file to .avi for video processing
        ravi_video_data.set(cv.CAP_PROP_FORMAT, -1)
        # changing the frame per seconds of the video
        ravi_video_data.set(cv.CAP_PROP_FPS, fps)
        # checking for the file format and raising error
        if not ravi_video_data.isOpened():
            raise Exception('Error while loading the {} video file'.format(self.Ravi_file_name))
        # extracting the height and width of the video 
        width = int(ravi_video_data.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(ravi_video_data.get(cv.CAP_PROP_FRAME_HEIGHT))
        f = 0
        print("Simulation Started \n")
        print('Progress: [', end='', flush=True)
        # opening the video for playing
        while ravi_video_data.isOpened() is True:
            # reading and fetching data for each frame
            fetch_status, frame = ravi_video_data.read()

            if fetch_status is False:
                print('] simulated ' + str(f) + ' time steps', end='', flush=True)
                print(' playing video is complete')
                break
            # resizing the frame for display
            re_frame = frame.view(dtype=np.int16).reshape(height, width)
            actual_frame = re_frame[1:, :]
            # To compensate the camera movement, the intensity peaks are identified and normalization is
            # performed for better visualization
            displace_frame = cv.normalize(actual_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            # Applying colormap for better visualization
            disp_color = cv.applyColorMap(displace_frame, cv.COLORMAP_JET)
            # Plotting each frame 
            cv.imshow('Optris RAVI file output', disp_color)
            #if f==950:
            #print(f)
            #    plt.imshow(displace_frame,cmap='RdYlBu_r', interpolation='None')
            #    plt.axis('off')
            #    plt.savefig(r"D:\STUDY_MATERIAL\document\optris_python"+str(f)+".png",dpi=600,bbox_inches='tight',transparent=True)
            cv.waitKey(10)

            #print(f)
            f += 1
            if f % 60 == 0:
                print('■', end='', flush=True)

        ravi_video_data.release()
        cv.destroyAllWindows()
        pass

    def ravi_to_yuv(self):
        """ Convert .ravi to yuv (binary file format) 
        """
        ravi_file_path = os.path.join(self.Root_directory, self.Ravi_file_name)
        yuv_file_name = self.Ravi_file_name[:-4] + "yuv"
        yuv_file_path = os.path.join(self.Root_directory, yuv_file_name)
        command = "ffmpeg -y -f avi -i '" + ravi_file_path + "' -vcodec rawvideo '" + yuv_file_path + "'"
        print(command)
        os.system(command)
        pass


if __name__ == '__main__':
    root_directory = r'utilites\datasets'
    experiment = r"2021-05-11 - Variantenvergleich - VarioTherm IR-Strahler - Winkel 45°"
    Vario = VarioTherm()
    temperature_data = Vario.image_sequence_extractor(root_dir, experiment, True)
    np.save(file_name + r'Documents/temp/temp1.npy', temperature_data)

    root_directory = r'utilites\datasets'
    video_file_name=r'experiment_1.ravi'
    a= Optris(root_directory,video_file_name)
    a.video_simulation()