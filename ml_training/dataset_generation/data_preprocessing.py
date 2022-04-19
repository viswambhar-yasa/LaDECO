# -*- coding: utf-8 -*-
## This file contain data preprocessing for object segmentation , thickness estimation and depth estimation
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermography videos)

#importing libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
# importing Utilities and thermal_profile classes from thermograms and utilites modules
from thermograms.Utilities import Utilities
from utilites.thermal_profile import thermal_profile


class data_preprocessing():
    """
    Generates datasets for machine learning module (Object segmentation)
    """
    def __init__(self) -> None:
        pass

    def data_filtering(self, experiment_data, scaling_type='normalization', index=1):
        """ This method filters data based on fourier transformation.
        The data belong to reflection phase which is required for object segmentation.

        Args:
            experiment_data (hadoop file format): Thermographic heat map data
            scaling_type (str, optional): Data standardization format type. Defaults to 'normalization'.
            index (int, optional): Initial index to remove the error time frames. Defaults to 1.

        Returns:
            _type_: filtered data
        """
        # convert hadoop file into numpy array for data manipulation
        data = np.array(experiment_data)
        # Applying Fast Fourier Transformation from numpy module equ. 5.3.1
        fourier_transformation = np.fft.fftn(data[:, :, index:-index], axes=(0, 1))
        # Calculating amplitude based on equ 5.3.2.
        amplitude = np.abs(fourier_transformation)
        # converting 3 dimensional array to 2 dimensional array using mean
        amplitude_sequence = np.mean(amplitude, axis=(0, 1))
        # calculating the step difference of the amplitude 
        delta_amplitude = np.diff(amplitude_sequence)
        ## Identifying reflection and radiation phase start index from the above equation
        sorted = np.sort(delta_amplitude)
        sorted_index = np.argsort(delta_amplitude)
        ## sorting the array to identify correct reflection phase removing error data
        if sorted_index[-1] > sorted_index[-2]:
            reflection_start_index = sorted_index[-2]
        else:
            reflection_start_index = sorted_index[-1]
        # reflection_start_index = np.argmax(delta_amplitude) + 5
        ## Adding tolerance value intial reflection start and end to correctly select the frames  
        reflection_start_index = reflection_start_index - 5
        reflection_end_index = np.argmin(delta_amplitude) - 5
        print("reflection_start_index: ",reflection_start_index," reflection_end_index: ", reflection_end_index)
        # Filtering experimental data based on reflection phase index
        if reflection_start_index > reflection_end_index:
            # print(5)
            filtered_data = data[:, :, 150:500 + 5]
        else:
            filtered_data = data[:, :, reflection_start_index:reflection_end_index]
        print(filtered_data.shape)
        # Normalizing data within a gaussian distribution to feed it to machine learning model
        if scaling_type == 'normalization':
            standardizing = StandardScaler()
            std_output_data = standardizing.fit_transform(
                filtered_data.reshape(filtered_data.shape[0], -1)).reshape(filtered_data.shape)
            return std_output_data
        # generating a gray scale image for thermograms data
        elif scaling_type == 'grayscale':
            gry_output_data = self.grayscale_image(filtered_data)
            return gry_output_data
        # Normalizing data within a -1 to 1  to feed it to machine learning model
        else:
            normalizing = MinMaxScaler(feature_range=(0, 1))
            nrm_output_data = normalizing.fit_transform(
                filtered_data.reshape(filtered_data.shape[0], -1)).reshape(filtered_data.shape)
            return nrm_output_data

    def create_training_dataset(self, dataset, experiment_list, annotaion_path, save=False, num_classes=4):
        """
        Creates training dataset by filtering reflecting input thermograms and creating segmentation from annotation path

        Args:
            dataset (Hadoop File format): Contains all thermographic video data 
            experiment_list (list): list of experiment name
            annotaion_path (str): path of annotations
            save (bool, optional): Value to save the thermogram and annotation data. Defaults to False.
            num_classes (int, optional): Segmentation classes. Defaults to 4.

        Returns:
            _type_ (numpy array): Input and output data
        """
        #initializing empty arrays for input and output dataset
        ouput_ds = np.array([])
        input_ds = np.array([])
        # run for all experiments in experiment list
        for i in experiment_list.values():
            exp_name = i
            print(exp_name)
            # extracting experiment data from hadoop file
            experimental_data = dataset[exp_name]
            ## Extrating reflected thermographic data for object segmentation
            filtered_ds = self.data_filtering(experimental_data)
            (m, n, num_of_ds) = filtered_ds.shape
            # Resizing the data into a standard format to feed into machine learning model
            filtered_data = np.resize(filtered_ds, new_shape=(256, 256, num_of_ds))
            # Extracting annotation from annotation path
            combined_mask = np.load(annotaion_path + i + '.npy')
            # Converting annotation mask by encoder to perform catgorical loss function
            combined_mask1 = tf.one_hot(combined_mask, depth=num_classes, axis=-1)
            ## repeating the encoded segmentation mask for each thermogram input
            dataset_mask = np.repeat(combined_mask1[:, :, :, np.newaxis], num_of_ds, axis=3)
            # creating an array to feed machine learning model
            if input_ds.shape[0] == 0:
                input_ds = filtered_data
                ouput_ds = dataset_mask
            else:
                input_ds = np.concatenate([input_ds, filtered_data], axis=-1)
                ouput_ds = np.concatenate([ouput_ds, dataset_mask], axis=-1)
            print("filtered dataset shape :", filtered_data.shape, "Shape after concatenation :", input_ds.shape)
        # Saving of input thermograms and output annotation 
        if save:
            with open('thermal_image_data.npy', 'wb') as f:
                np.save(f, input_ds)
            with open('annotation_data.npy', 'wb') as f:
                np.save(f, ouput_ds)
        return input_ds, ouput_ds

    def train_test_split_index(self, number_of_time_steps):
        """
        creates splits index based on time steps to generate training, validation and test datasets

        Args:
            number_of_time_steps (init): timestep based on which the data is seperated

        Returns:
            _type_: return training, validation and test index 
        """
        # generate a list of arrays base
        time_index = np.arange(0, number_of_time_steps, step=1)
        time_index = np.arange(0, number_of_time_steps, step=1)
        ## splitting time frames to generate training and validation time frames based on sklearn 
        train_index, val_index = train_test_split(time_index, test_size=0.3)
        ## splitting validation frames to generate test time framesvbased on sklearn module
        validation_index, test_index = train_test_split(val_index, test_size=0.3)
        return train_index, validation_index, test_index

    def load_image_segmentation_dataset(self, dataset, experiment_list, annotaion_path, save=False, num_classes=4):
        """
        This method combines the above methods to create datasets for training
        Args:
            dataset (Haddop file): Big data file containing thermographic 
            experiment_list (list): list of experiments in the hadoop file
            annotaion_path (str): Path of the annotation files
            save (bool, optional): To save input and output data for each experiment. Defaults to False.
            num_classes (int, optional): Segmentation classes. Defaults to 4.

        Returns:
            _type_: Creating training, validation and test dataset
        """
        # creating input thermogram and output segmentation mask by filtering reflection phase data for all experiment in the hadoop file
        input_images, output_annotations = self.create_training_dataset(dataset, experiment_list, annotaion_path,
                                                                        save=True)
        # Generating time index for training, validation and test 
        if input_images.shape[2] == output_annotations.shape[3]:
            number_of_time_steps = input_images.shape[2]
            train_index, validation_index, test_index = self.train_test_split_index(number_of_time_steps)
        # splitting data based on time index from train test split
        train_input = input_images[:, :, train_index]
        train_output = output_annotations[:, :, :, train_index]
        val_inputs = input_images[:, :, validation_index]
        val_ouputs = output_annotations[:, :, :, validation_index]
        test_inputs = input_images[:, :, test_index]
        test_ouputs = output_annotations[:, :, :, test_index]
        return (train_input, train_output), (val_inputs, val_ouputs), (test_inputs, test_ouputs)
        ## return (tf.expand_dims(train_input,axis=-1), tf.expand_dims(train_output,axis=-1)), (tf.expand_dims(val_inputs,axis=-1), tf.expand_dims(val_ouputs,axis=-1)), (tf.expand_dims(test_inputs,axis=-1), tf.expand_dims(test_ouputs,axis=-1))

    def create_dataset(self, input_image, input_segementation, BATCH_SIZE=32, num_classes=3):
        """ 
        This method convert numpy array into tensor object

        Args:
            input_image (numpy array): thermogram in spatial domain
            input_segementation (_type_): segementation masks of thermograms
            BATCH_SIZE (int, optional): Batch size. Defaults to 32.
            num_classes (int, optional): segmentation classes. Defaults to 3.

        Returns:
            _type_: _description_
        """
        dataset_input = []
        dataset_output = []
        for i in range(input_segementation.shape[3]):
            dataset_input.append(tf.expand_dims(input_image[:, :, i], axis=-1))
            dataset_output.append(input_segementation[:, :, :, i])
        dataset_input = np.array(dataset_input)
        dataset_output = np.array(dataset_output)
        dataset = tf.data.Dataset.from_tensor_slices((dataset_input, dataset_output))
        return dataset


class data_preprocessing1():
    """
    Similar to the above class but used for ascii dataset
    """
    def __init__(self) -> None:
        pass

    def data_filtering(self, experiment_data, scaling_type='normalization', index=1):
        data = np.array(experiment_data)
        fourier_transformation = np.fft.fftn(data[:, :, index:], axes=(0, 1))
        amplitude = np.abs(fourier_transformation)
        amplitude_sequence = np.mean(amplitude, axis=(0, 1))
        delta_amplitude = np.diff(amplitude_sequence)

        sorted = np.sort(delta_amplitude)
        sorted_index = np.argsort(delta_amplitude)
        if sorted_index[-1] > sorted_index[-2]:
            reflection_start_index = sorted_index[-2]
        else:
            reflection_start_index = sorted_index[-1]
        reflection_end_index = np.argmin(delta_amplitude) - 5
        print(reflection_start_index, reflection_end_index)
        if reflection_start_index > reflection_end_index:
            # print(5)
            filtered_data = data[:, :, 150:500 + 5]
        else:
            filtered_data = data[:, :, reflection_start_index:reflection_end_index]
        print(filtered_data.shape)
        if scaling_type == 'normalization':
            standardizing = StandardScaler()
            std_output_data = standardizing.fit_transform(
                filtered_data.reshape(filtered_data.shape[0], -1)).reshape(filtered_data.shape)
            return std_output_data
        elif scaling_type == 'grayscale':
            gry_output_data = self.grayscale_image(filtered_data)
            return gry_output_data
        else:
            normalizing = MinMaxScaler(feature_range=(0, 1))
            nrm_output_data = normalizing.fit_transform(
                filtered_data.reshape(filtered_data.shape[0], -1)).reshape(filtered_data.shape)
            gry_output_data = self.grayscale_image(nrm_output_data)
            return nrm_output_data

    def create_training_dataset(self, dataset, experiment_list, annotaion_path, save=False, num_classes=4):
        ouput_ds = np.array([])
        input_ds = np.array([])
        for i in experiment_list.values():
            exp_name = i
            print(exp_name)
            experimental_data = dataset[exp_name]
            filtered_ds = self.data_filtering(experimental_data)
            (m, n, num_of_ds) = filtered_ds.shape
            filtered_data = np.resize(filtered_ds, new_shape=(256, 256, num_of_ds))
            combined_mask1 = np.load(annotaion_path + i + '.npy')
            combined_mask = tf.one_hot(combined_mask1, depth=num_classes, axis=-1)
            dataset_mask = np.repeat(combined_mask[:, :, :, np.newaxis], num_of_ds, axis=3)
            if input_ds.shape[0] == 0:
                input_ds = filtered_data
                ouput_ds = dataset_mask
            else:
                input_ds = np.concatenate([input_ds, filtered_data], axis=-1)
                ouput_ds = np.concatenate([ouput_ds, dataset_mask], axis=-1)
            print("filtered dataset shape :", filtered_data.shape, "Shape after concatenation :", input_ds.shape)
        if save:
            with open('thermal_image_data.npy', 'wb') as f:
                np.save(f, input_ds)
            with open('annotation_data.npy', 'wb') as f:
                np.save(f, ouput_ds)
        return input_ds, ouput_ds

    def train_test_split_index(self, number_of_time_steps):
        time_index = np.arange(0, number_of_time_steps, step=1)
        time_index = np.arange(0, number_of_time_steps, step=1)
        train_index, val_index = train_test_split(time_index, test_size=0.3)
        validation_index, test_index = train_test_split(val_index, test_size=0.3)
        return train_index, validation_index, test_index

    def load_image_segmentation_dataset(self, dataset, experiment_list, annotaion_path, save=False, num_classes=3):
        input_images, output_annotations = self.create_training_dataset(dataset, experiment_list, annotaion_path,
                                                                        save=True)
        if input_images.shape[2] == output_annotations.shape[3]:
            number_of_time_steps = input_images.shape[2]
            train_index, validation_index, test_index = self.train_test_split_index(number_of_time_steps)
        train_input = input_images[:, :, train_index]
        train_output = output_annotations[:, :, :, train_index]
        val_inputs = input_images[:, :, validation_index]
        val_ouputs = output_annotations[:, :, :, validation_index]
        test_inputs = input_images[:, :, test_index]
        test_ouputs = output_annotations[:, :, :, test_index]
        return (train_input, train_output), (val_inputs, val_ouputs), (test_inputs, test_ouputs)
        # return (tf.expand_dims(train_input,axis=-1), tf.expand_dims(train_output,axis=-1)), (tf.expand_dims(val_inputs,axis=-1), tf.expand_dims(val_ouputs,axis=-1)), (tf.expand_dims(test_inputs,axis=-1), tf.expand_dims(test_ouputs,axis=-1))

    def create_dataset(self, input_image, input_segementation, BATCH_SIZE=32, num_classes=3):
        dataset_input = []
        dataset_output = []
        for i in range(input_segementation.shape[3]):
            dataset_input.append(tf.expand_dims(input_image[:, :, i], axis=-1))
            dataset_output.append(input_segementation[:, :, :, i])
        dataset_input = np.array(dataset_input)
        dataset_output = np.array(dataset_output)
        dataset = tf.data.Dataset.from_tensor_slices((dataset_input, dataset_output))
        return dataset


def thickness_data_preprocessing(thermal_data, experiment_list, tol=5, start_tol=10, constrast_val=271, x_crop_i=35,
                                 y_crop_i=25, x_crop_j=20, y_crop_j=35, no_of_time_steps=200, disp=True):
    """
    This function generates thickness estimation dataset

    Args:
        thermal_data (hadoop file): Thermographic video file 
        experiment_list (list ): lsit of experiments 
        tol (int, optional): Value added to reflection start and end to get exact index. Defaults to 5.
        start_tol (int, optional): initial tolerance to filter thermographic data to remove errors. Defaults to 10.
        constrast_val (int, optional): Value to calculate contrast function. Defaults to 271.
        x_crop_i (int, optional): initial cropping index for width. Defaults to 35.
        y_crop_i (int, optional): final cropping index for width. Defaults to 25.
        x_crop_j (int, optional): initial cropping index for height. Defaults to 20.
        y_crop_j (int, optional): final cropping index for height. Defaults to 35.
        no_of_time_steps (int, optional): length of radiation phase to be extracted . Defaults to 200.
        disp (bool, optional): For printing data and plots. Defaults to True.

    Returns:
        _type_: Training data containing the radiation phase after performing constrast value
    """
    # initializing empty dataset 
    training_data = []
    # running for all experiment 
    for experiment in experiment_list.values():
        print(experiment)
        # extracting experiment data from hadoop file 
        experimental_data = thermal_data[experiment]
        # converting hadoop file into numpy array
        data = np.array(experimental_data)
        # converting temperature input into fourier signal using fast fourier module in numpy 
        fourier_transformation = np.fft.fftn(data, axes=(0, 1))
        # Calculating amplitude based on equ 5.3.2.
        amplitude = np.abs(fourier_transformation)
        # converting 3 dimensional array to 2 dimensional array using mean
        amplitude_sequence = np.mean(data, axis=(0, 1))
        # caluating step difference to identify relfection phase
        delta_amplitude = np.diff(amplitude_sequence)
        ## print(np.sort(delta_amplitude)[-5:])
        ## print(np.argsort(delta_amplitude)[-5:])
        # identifying reflection phase index 
        # sorting the array to identify correct reflection phase removing error data
        sorted = np.sort(delta_amplitude)
        sorted_index = np.argsort(delta_amplitude)
        if sorted_index[-1] > sorted_index[-2]:
            reflection_start_index = sorted_index[-2]
        else:
            reflection_start_index = sorted_index[-1]
        # Adding tolerance value for intial reflection start and end to correctly select the frames
        ## reflection_start_index = np.argmax(delta_amplitude) + 5
        reflection_start_index = reflection_start_index - tol
        reflection_end_index = np.argmin(delta_amplitude[:-5]) - tol
        print('reflection_start_index:', reflection_start_index, 'reflection_end_index:', reflection_end_index)

        # Cropping the initial thermogram to get effective area where heat is incident
        input_data = data[x_crop_i:-y_crop_i, x_crop_j:-y_crop_j, :]
        # calculating the mean temperature till the reflection phase
        based_temp = np.mean(input_data[:, :, start_tol:reflection_start_index - start_tol])
        ## avg_data= input_data[:,:,reflection_end_index-20]
        ## input_data=input_data[:,:,reflection_start_index+7:]
        ## standardizing = MinMaxScaler(feature_range=(0, 1))
        ## std_output_data = standardizing.fit_transform(
        ##              input_data.reshape(input_data.shape[0], -1)).reshape(input_data.shape)
        # performing contrasting operation to get input data for thickness estimation
        id_data = (input_data[:, :,
                   reflection_end_index + start_tol:reflection_end_index + start_tol + no_of_time_steps] - constrast_val) / constrast_val
        ## reflection_length=(reflection_end_index-reflection_start_index)
        ## normalized_data=std_output_data[:,:,reflection_length+10:reflection_length+10+no_of_time_steps]
        # converting 3d data into 2d 
        mean_inp = np.mean(id_data, axis=(0, 1))
        
        ## plt.legend(exp_list2,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        training_data.append(tf.expand_dims(tf.expand_dims(mean_inp, axis=0), axis=0))
    # to print the shape and plot the contrast function for all experiments
    if disp:
        print(mean_inp.shape)
        plt.plot(mean_inp)
        ## plt.savefig("/content/thermal_coating_thickness.png",dpi=600,bbox_inches='tight',transparent=True)
    # changing the dimension of the data which can be feed to a machine learning model
    x_train_ds = tf.concat(training_data, axis=0)
    return x_train_ds


def data_preprocessing_depth(thermal_data, experiment_list, anno_path, tol=5, features=3, index=1, no_of_timesteps=200):
    """
    This function generate depth estimation datasets

    Args:
        thermal_data (Hadoop data file): Contains all thermographic experiments
        experiment_list (list): list of experiment in the hadoop file
        anno_path (str): path of the folder containing segmentation masks
        tol (int, optional): Tolerance to identify reflection phase. Defaults to 5.
        features (int, optional): Number of feature in the segmentation mask. Defaults to 3.
        index (int, optional): Index to remove the error thermograms in the thermographic experiments. Defaults to 1.
        no_of_timesteps (int, optional): The length of the radiation phase. Defaults to 200.

    Returns:
        _type_: Training data input 
    """
    # number of experiments in the hadoop file
    n = len(experiment_list)
    # creating containers based on number features
    x_train_ds_3 = np.zeros(shape=[n, no_of_timesteps, 3])
    x_train_ds_2 = np.zeros(shape=[n, no_of_timesteps, 2])

    i = 0
    for experiment in experiment_list.values():
        print(experiment)
        # extracting specific experiment for hadoop file
        experimental_data = thermal_data[experiment]
        #converting it into numpy array
        data = np.array(experimental_data)
        # performing fast fourier transformation based on equ 5.3.1
        fourier_transformation = np.fft.fftn(data[:, :, index:-index], axes=(0, 1))
        # Calculating the amplitude based on equ. 5.3.2
        amplitude = np.abs(fourier_transformation)
        # converting 3d data into 2d sequence for analysis
        amplitude_sequence = np.mean(amplitude, axis=(0, 1))
        # calculating step difference to identify peaks changes in the data
        delta_amplitude = np.diff(amplitude_sequence)
        # sorting the data to identify the phases
        sorted = np.sort(delta_amplitude)
        sorted_index = np.argsort(delta_amplitude)
        if sorted_index[-1] > sorted_index[-2]:
            reflection_start_index = sorted_index[-2]
        else:
            reflection_start_index = sorted_index[-1]
        ## reflection_start_index = np.argmax(delta_amplitude) + 5
        # Adding a tolerance vaue to the initial phase to correct the reflection phase
        reflection_start_index = reflection_start_index - tol
        reflection_end_index = np.argmin(delta_amplitude) + tol
        print('reflection_start_index: ', reflection_start_index, ' reflection_end_index: ', reflection_end_index)
        # importing segmentation annotation
        annotation_path = anno_path + experiment + '.npy'
        combined_mask = np.load(annotation_path)
        # performing masking operation on each thermogram to extrast thermal profile of all features
        substrate, background, defects, thermal_band = thermal_profile(experimental_data, combined_mask,
                                                                       feature=features)
        # applying constrast function based on equ. 6.3.1                                                               
        constrast_sub = (np.array(substrate) - np.array(background)) / np.array(background)
        constrast_def = (np.array(defects) - np.array(background)) / np.array(background)
        constrast_thermal = (np.array(thermal_band) - np.array(background)) / np.array(background)
        # loading constrast function of all features
        if features == 3:
            x_train_ds_3[i, :, 0] = np.squeeze(
                constrast_sub[reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps])
            x_train_ds_3[i, :, 1] = np.squeeze(
                constrast_def[reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps])
            x_train_ds_3[i, :, 2] = np.squeeze(
                constrast_thermal[reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps])
            i += 1
        else:
            x_train_ds_2[i, :, 0] = constrast_sub[
                                    reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps]
            x_train_ds_2[i, :, 1] = constrast_def[
                                    reflection_end_index + tol:reflection_end_index + tol + no_of_timesteps]
            i += 1
    # converts any null or NAN to 0
    if features == 3:
        x_train_ds_3[np.isnan(x_train_ds_3)] = 0
        return x_train_ds_3
    else:
        x_train_ds_2[np.isnan(x_train_ds_2)] = 0
        return x_train_ds_2


def depth_annotations(experiment_list, thickness_list, feature=3):
    """
    This function generates output data for depth estimation process
    Args:
        experiment_list (list): list of experiments
        thickness_list (list): contains the thickness or depth of all features for all experiment 
        feature (int, optional): Number of segmentation features. Defaults to 3.

    Returns:
        _type_: Output data 
    """
    # number of experiment in the experiment list 
    n = len(experiment_list)
    # creating container for storing data
    y_train = np.zeros(shape=(n, feature))
    index = 0
    # loading thickness data of respective feature into container
    # 0- substrate
    # 1- damaged substrate
    # 2- thermal band
    for i in range(0, n):
        if feature == 3:
            y_train[i, 0] = thickness_list[index]
            index += 1
            y_train[i, 1] = thickness_list[index]
            index += 1
            y_train[i, 2] = thickness_list[index]
            index += 1
        else:
            y_train[i, 0] = thickness_list[index]
            index += 1
            y_train[i, 1] = thickness_list[index]
            index += 1
    # changing the dimension to enable data to be feed to a machine learning model        
    y_train_ds = np.expand_dims(y_train, axis=0)
    return y_train_ds


if __name__ == '__main__':
    # sample code to check the training data
    root_path = r'utilites/datasets'
    data_file_name = r'Variantenvergleich_data_python_api.hdf5'
    a = Utilities()
    thermal_data, experiment_list = a.open_file(root_path, data_file_name, True)
    experiment_name = r'2021-12-15-Materialstudie_Metallproben-ML1-laserbehandelte_Probe-150W-10s'
    annotation_path = r'ml_training\dataset_generation\masks\final_masks'
    # exp_list={key: value for key, value in experiment_list.items() if key in [0,1,2]}
    dataset_genertor = data_preprocessing()
    train_ds, valid_ds, test_ds = dataset_genertor.load_image_segmentation_dataset(thermal_data, experiment_list,
                                                                                   annotation_path, save=True)
    # training_dataset,validation_dataset,testing_dataset=dataset_genertor.load_image_segmentation_dataset(thermal_data,experiment_list,annotation_path)
    print(train_ds[0].shape, train_ds[1].shape)
    training_dataset = dataset_genertor.create_dataset(train_ds[0], train_ds[1])
    validation_dataset = dataset_genertor.create_dataset(np.array(valid_ds[0]), valid_ds[1])
    test_dataset = dataset_genertor.create_dataset(test_ds[0], test_ds[1])
    # training_dataset = aug_train_ds.batch(batch_size)
    batch_size = 32
    training_dataset = training_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    # validation_dataset = new_validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    len(training_dataset), len(validation_dataset), len(test_dataset)
    segementation_model = tf.keras.models.load_model(
        r"trained_models\UNet\u_net_64_Adam_5_best_agu.h5")
    segementation_model.summary()
    testdata = next(iter(training_dataset))
    predict = segementation_model.predict(testdata[0])
    print(predict.shape)
    i = 25
    predicted = predict[i, :, :, :]
    image = testdata[0][i]
    predicted_anno = testdata[1][i]
    # agumented=data_augmentation(image)
    # plt.imshow(np.squeeze(image))
    anno = np.zeros((256, 256))
    for i in range(4):
        temp = predicted_anno[:, :, i].numpy()
        anno[temp == 1] += i
    predicted[predicted > 0.5] = 1
    predicted[predicted < 0.5] = 0
    predicted = np.squeeze(predicted)
    pre_anno = np.zeros((256, 256))
    for i in range(4):
        temp = predicted[:, :, i]
        pre_anno[temp == 1] += i

    # create figure
    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(1, 3, 1)
    # showing image
    plt.imshow(np.squeeze(image), cmap='gray')
    # plt.axis('off')
    plt.title("image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 3, 2)

    # showing image
    plt.imshow(anno, cmap='gray')
    # plt.axis('off')
    plt.title("ground truth")

    # Adds a subplot at the 3rd position
    fig.add_subplot(1, 3, 3)

    # showing image
    plt.imshow(pre_anno, cmap='gray')
    # plt.axis('off')
    plt.title("predicted mask")
    plt.show()
