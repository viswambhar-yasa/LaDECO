# -*- coding: utf-8 -*-
## This file contain classes to build machine learning models for object segmentation, thickness estimation and depth estimation
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermography videos)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,concatenate,Activation,UpSampling2D,GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2DTranspose, Concatenate, \
    Input, AveragePooling2D, Cropping2D, Add,ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import numpy as np


class U_Net():
    """
    Object segmentation model U_net built using on U-Net: Convolutional Networks for Biomedical Image Segmentation
    by Olaf Ronneberger, Philipp Fischer, Thomas Brox
    """
    def __init__(self) -> None:
        pass

    def convolution_encoder(self, inputs, cn_filters, cn_kernel_size, cn_activation="relu", drop_out_prob=0.2,
                            pooling_type="Max", pool=True, drop=False, batch=False) -> np.ndarray:
        """
        Basic block of convolution encoder which conist of convolution operation, normalization and pooling

        Args:
            inputs (tensor): layer input 
            cn_filters (int): Number of times the convolution operation 
            cn_kernel_size (int): The convolution window size 
            cn_activation (str, optional): Activation function . Defaults to "relu".
            drop_out_prob (float, optional): Regularization method by killing neuron . Defaults to 0.2.
            pooling_type (str, optional): Pooling operation to reduce the parameters and noise. Defaults to "Max".
            pool (bool, optional): parameter to avoid pooling operation. Defaults to True.
            drop (bool, optional): parameter to avoid drop operation. Defaults to False.
            batch (bool, optional): parameter to avoid batch normalization operation. Defaults to False.

        Returns:
            tensor: output of convolution block
        """
        # 2D convolution operation with respective filters, activation function and kernel size 
        x = Conv2D(filters=cn_filters,
                   kernel_size=cn_kernel_size, activation=cn_activation, padding='same',
                   kernel_initializer='he_normal')(inputs)
        if batch:
            # Batch normalization operation (to reduce normalize the data)
            x = BatchNormalization()(x)
        if drop:
            # Drop out operation (to reduce overfitting)
            x = Dropout(drop_out_prob)(x)
        # Pooling layer (to reduce noise)
        if pooling_type == "Max" and pool:
            p = MaxPooling2D(pool_size=(2, 2))(x)
            return p, x
        elif pooling_type == "Avg" and pool:
            p = AveragePooling2D(pool_size=(2, 2))(x)
            return p, x
        else:
            return x

    def convoluton_decoder(self, inputs, conv_inputs, filters, kernel_size=(3, 3), activation="relu", drop_out_prob=0.2,
                           drop=True):
        """
        Later part of the U-net which performs decoder or upsampling of the data

        Args:
            inputs (tensor): layer input
            conv_inputs (tensors): encoder data obtained during downsampling
            filters (int): Number of filter in each convolution layer
            kernel_size (tuple, optional): The size of the convolution window. Defaults to (3, 3).
            activation (str, optional): Activation function. Defaults to "relu".
            drop_out_prob (float, optional): Regularization method by killing neuron  . Defaults to 0.2.
            drop (bool, optional): parameter to avoid batch normalization operation . Defaults to True.

        Returns:
            tensor: ouput of decoder block
        """
        # Transpose of convolution operation (upsampling of data)
        x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(
            2, 2), activation=activation, padding='same', kernel_initializer='he_normal')(inputs)
        # Concatenation of upsampled data and encoder output    
        concat = Concatenate(axis=-1)([x, conv_inputs])
        # Performing convolution operation for respective filters, activation and kernel size
        x = Conv2D(filters, kernel_size, activation=activation,
                   padding='same', kernel_initializer='he_normal')(concat)
        if drop:
            # Drop out operation (to reduce overfitting)
            x = Dropout(drop_out_prob)(x)
        # Performing convolution operation for respective filters, activation and kernel size
        output = Conv2D(filters, kernel_size, activation=activation,
                        padding='same', kernel_initializer='he_normal')(x)
        return output

    def create_model(self, num_classes, input_shape, filters_list, model_config=None, disp=True):
        """
        This method combines encoder, decoder and create a U net model
        Args:
            num_classes (_type_): Signifies the number of encoder and decoder blocks
            input_shape (_type_): The shape of the input image
            filters_list (_type_): the number of filters in each encoder and decoder block
            model_config (_type_, optional): . Defaults to None.
            disp (bool, optional): parameter to plot the shapes of each block and model archiecture. Defaults to True.

        Returns:
            _type_: U Net model
        """
        # Creating input layer with shape of input shape
        pipe_line_inputs = Input(input_shape)
        print("Building U-NET model \n")
        print('Input shape :', pipe_line_inputs.shape)
        pooled = pipe_line_inputs
        encoder_cnn_list = {}
        # creating a encoder block 
        for i in range(len(filters_list) - 1):
            # extracting filters for each encoder block
            filters = filters_list[i]
            pooled, x = self.convolution_encoder(
                inputs=pooled, cn_filters=filters, cn_kernel_size=(3, 3), drop=True)
            encoder_cnn_list[i] = x
        encoder_output = pooled
        # this layer links the encoder and decoder
        print("pipline encoder output shape:", encoder_output.shape)
        bottle_neck = self.convolution_encoder(
            encoder_output, cn_filters=filters_list[-1], cn_kernel_size=(3, 3), drop_out_prob=0.1, drop=True,
            pool=False)
        print("pipline bottle-neck output shape:", bottle_neck.shape)
        x = bottle_neck
        # creating a decoder block 
        for i in reversed(range(len(filters_list) - 1)):
            filters = int(filters_list[i])
            encoder_cnn = encoder_cnn_list[i]
            x = self.convoluton_decoder(
                inputs=x, conv_inputs=encoder_cnn, filters=filters)
        decoder_output = x
        print("pipline decoder output shape:", decoder_output.shape)
        # final convolution layer with is of the size the number of features the U-Net need to identify
        pipe_line_output = Conv2D(num_classes, kernel_size=(
            1, 1), activation='softmax')(decoder_output)
        print("output shape", pipe_line_output.shape, '\n')
        # Creating model using input and ourput layer
        model = Model(inputs=pipe_line_inputs, outputs=pipe_line_output)
        print('Model built sucessfully \n')
        if disp:
            # Plots model archiecture 
            tf.keras.utils.plot_model(
                model, to_file='u_net_arch.png', show_shapes=True)
            model.summary()
        return model


class FCN8():
    """
    Fully convolution Network used for object segmentation built using 
    Supervised Classification of Multisensor Remotely Sensed Images Using a Deep Learning Framework
    """
    def __init__(self) -> None:
        """initial parameters required to build the network
        """
        self.image_shape = (256, 256, 3)
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.pool_stride = (2, 2)
        self.activation = 'relu'
        self.filter_list = [64, 128, 256, 512, 512]
        self.layers_list = [2, 2, 3, 3, 3]
        self.n = 4096
        pass

    def convolution_block(self, block_input, num_of_layers, filters, kernel_size, pool_size, pool_stride, activation,
                          block_name):
        """creates convolution blocks which consist of convlution operation, pooling, batch normalization

        Args:
            block_input (tensor): input to the block
            num_of_layers (int): number of convolution layers
            filters (list): filters of each layer
            kernel_size (int): size of convolution window
            pool_size (int): pooling window size
            pool_stride (int): pooling window stide or skip
            activation (str): Activation function type
            block_name (str): Name of the convolution block

        Returns:
            _type_: convolution block
        """
        # creating a convolution block with the required convolution layer
        for i in range(num_of_layers):
            # 2d convolution layer
            block_ouput = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
                                 name="block{}_conv{}".format(block_name, i + 1))(block_input)
            block_input = block_ouput
        # Max pooling (to reduce noise)
        output = MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same',
                              name="block{}_pool".format(block_name))(block_input)
        return output

    def image_encoder(self, image_input, weights_path=None, disp=False):
        """
        Builts encoder layer with multiple convolution blocks
        Args:
            image_input (_type_): input to the layer
            weights_path (_type_, optional): Pre trained model weights. Defaults to None.
            disp (bool, optional): Parameter for plotting. Defaults to False.

        Returns:
            _type_: Encoder block
        """
        print('Building CNN encoder')
        encoder_ouput = {}
        encoder_ouput[0] = image_input
        x = image_input
        # building convolution block based on the size of filter list
        for i, filters in enumerate(self.filter_list):
            x = self.convolution_block(block_input=x, num_of_layers=self.layers_list[i], filters=filters,
                                       kernel_size=self.kernel_size, pool_size=self.pool_size,
                                       pool_stride=self.pool_stride, activation='relu', block_name=str(i + 1))
            encoder_ouput[i + 1] = x

        x = Conv2D(self.n, (7, 7), activation='relu', padding='same', name="conv6")(x)
        image_encoder_ouput = Conv2D(self.n, (1, 1), activation='relu', padding='same', name="conv7")(x)
        encoder_ouput[i + 2] = image_encoder_ouput
        # print(encoder_ouput)
        model_output = encoder_ouput[len(self.filter_list)]
        print('CNN encoder {output shape}', image_encoder_ouput.shape)
        # creating a encoder model 
        encoder_model = Model(inputs=image_input, outputs=model_output, name='based_model')
        if weights_path is not None:
            encoder_model.load_weights(weights_path)
        if disp:
            tf.keras.utils.plot_model(encoder_model, to_file='pre_trained_model_FC9.png', show_shapes=True)
            encoder_model.summary()
        return encoder_ouput

    def fc8_decoder(self, image_input, num_classes, weights_path=None):
        """performs the required operations encoder layer to build the FC8 model 

        Args:
            image_input (tensor): Input to the layer
            num_classes (int): Number of features for segmentation
            weights_path (hd5, optional): Pre-trained weight to optimized training. Defaults to None.

        Returns:
            _type_: _description_
        """
        # generating image encoder and extracting the ouput of each convolution block
        encoder_ouput = self.image_encoder(image_input, weights_path)
        print('Building Decoder')
        # Specific upsampling to build the model
        x = encoder_ouput[6]
        y = Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2))(x)
        y = Cropping2D(cropping=(1, 1))(y)

        x2 = encoder_ouput[4]
        y2 = Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x2)

        x3 = Concatenate([y, y2], axis=-1)

        y3 = Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2))(x3)
        y3 = Cropping2D(cropping=(1, 1))(y3)

        x4 = encoder_ouput[3]
        y4 = Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x4)

        x5 = Concatenate([y3, y4], axis=-1)
        # creating a decoder model
        decoder_output = Conv2DTranspose(num_classes, (8, 8), strides=(8, 8), activation='softmax')(x5)
        print('Decoder {Output shape}', decoder_output.shape)
        return decoder_output

    def auto_encoder(self, weights_path=None, num_classes=2, image_shape=(256, 256, 1), disp=False):
        """
        Encoder + Decoder is called auto decoder

        Args:
            weights_path (_type_, optional): weight for encoder layer. Defaults to None.
            num_classes (int, optional): Number of segmentation features. Defaults to 2.
            image_shape (tuple, optional): Shape of the input image. Defaults to (256, 256, 1).
            disp (bool, optional): Parameter to plot shapes of each layer and model archiecture. Defaults to False.

        Returns:
            _type_: _description_
        """
        print('Building a Fully Connected convolution network')
        model_input = Input(shape=image_shape)
        # building the model to match the pre trained weights 
        if weights_path is None:
            print('Input shape', model_input.shape)
            model_ouput = self.fc8_decoder(model_input, num_classes, weights_path)
        else:
            image_con = Concatenate()([model_input, model_input, model_input])
            print('Input shape', image_con.shape)
            model_ouput = self.fc8_decoder(image_con, num_classes, weights_path)
        segmentation_model = Model(inputs=model_input, outputs=model_ouput, name='segmentation_model')
        print('Model built successfully')
        if disp:
            # plotting the model archiecture
            tf.keras.utils.plot_model(segmentation_model, to_file='segmentation_FC9.png', show_shapes=True)
            segmentation_model.summary()
        return segmentation_model

class FCN():
  def __init__(self,no_classes,):
    self.Classes=no_classes

  def VGG_encoder(self,input_img,filter_list,layers_list):
    block_name='block'
    block_input=input_img
    #data_dic={}
    #data_dic['image_input']=block_input
    for j,filters in enumerate(filter_list):
      for i in range(layers_list[j]):
        block_ouput=Conv2D(filters=filters,kernel_size=(3,3),activation='relu',padding='same',name="block{}_conv{}".format(j+1,i+1))(block_input)
        block_input=block_ouput
      #print(j)
      pool_output=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name="block{}_pool".format(j+1))(block_input) 
      block_input=pool_output 
      #data_dic['f'+str(j+1)]=output
    vgg_model=Model(inputs=input_img,outputs=pool_output,name='vgg_encoder')
    return vgg_model

  def FCN_decoder(self,image_shape=(256,256,1),filter_list=[64,128,256,512,512],layers_list=[2,2,3,3,3],weights_path=None,disp=True):
    input_img=Input(shape=image_shape)
    if weights_path is not None:
      input_img=Conv2D(3,kernel_size=(3,3))(input_img)
      vgg_encoder=self.VGG_encoder(input_img,filter_list=filter_list,layers_list=layers_list)
      vgg_encoder.load_weights(weights_path)
    else:
      vgg_encoder=self.VGG_encoder(input_img,filter_list=filter_list,layers_list=layers_list)
    x=vgg_encoder.get_layer('block'+str(len(filter_list))+'_pool').output
    y=Conv2DTranspose(self.Classes,kernel_size=(4,4),strides=(2,2))(x)
    y=Cropping2D(cropping=(1,1))(y)

    x2=vgg_encoder.get_layer('block'+str(len(filter_list)-1)+'_pool').output
    y2=Conv2D(self.Classes,(1,1),activation='relu',padding='same')(x2)
    
    x3=concatenate([y,y2],axis=-1)

    y3=Conv2DTranspose(self.Classes,kernel_size=(4,4),strides=(2,2))(x3)
    y3=Cropping2D(cropping=(1,1))(y3)
    
    x4=vgg_encoder.get_layer('block'+str(len(filter_list)-2)+'_pool').output
    y4=Conv2D(self.Classes,(1,1),activation='relu',padding='same')(x4)
    
    x5=concatenate([y3,y4],axis=-1)
    
    decoder_output=Conv2DTranspose(self.Classes,(8,8),strides=(8,8),activation='softmax')(x5)

    FCN_model=Model(inputs=input_img,outputs=decoder_output,name='segmentation_model')
    if disp:
      tf.keras.utils.plot_model(FCN_model,show_shapes=True)
    return FCN_model
    
class PSP_Net():

  def __init__(self):
        pass
  
  def convolution_block(self,conv_input,filter,block_index):
    """
      Create a convolution block consisting of convolution, pooling and normalization layer

      Args:
          conv_input (tensor): input to the block
          filter (list): conatins the filter sizes of the convolution layer
          block_index (int): Used for naming the convolution block

      Returns:
          tensor: convolution block
    """
    block_name='Layer'+block_index+'_'
    # convolution layers 1 followed by normalization
    x=Conv2D(filters=filter[0],kernel_size=(1,1),dilation_rate=(1,1),activation='relu',padding='same',kernel_initializer='he_normal',name=block_name+'A')(conv_input)
    x1=BatchNormalization(name=block_name+'batch_norm_A')(x)
    # convolution layers 2 followed by normalization
    y=Conv2D(filters=filter[1],kernel_size=(3,3),dilation_rate=(2,2),activation='relu',padding='same',kernel_initializer='he_normal',name=block_name+'B')(x1)
    y1=BatchNormalization(name=block_name+'batch_norm_B')(y)
    # convolution layers 3 followed by normalization
    z=Conv2D(filters=filter[2],kernel_size=(1,1),dilation_rate=(1,1),activation='relu',padding='same',kernel_initializer='he_normal',name=block_name+'C')(y1)
    z1=BatchNormalization(name=block_name+'batch_norm_C')(z)
    # convolution layer of the first layer skipping layer 2 and 3 
    skip_conv=Conv2D(filters=filter[2],kernel_size=(3,3),dilation_rate=(2,2),activation='relu',padding='same',name=block_name+'skip_conv')(x1)
    Z=BatchNormalization(name=block_name+'batch_norm_skip')(skip_conv)
    # adding the skipped layer and normal convolution layer
    added_layers = Add(name=block_name+'add')([z1,Z])
    # applying activition function
    conv_blk_output = ReLU(name=block_name+'relu')(added_layers)
    return conv_blk_output

  def feature_map_extraction(self,input_layer):
    """
      Generate convolution block of different sizes 

      Args:
          input_layer (tensor): Input to th elayer

      Returns:
          _type_: _description_
    """
    # convolution blocks of different filters sizes  
    block1 = self.convolution_block(input_layer,[16,16,32],'1')
    block2 = self.convolution_block(block1,[32,32,64],'2')
    block3 = self.convolution_block(block2,[64,64,128],'3')
    return block3
    
  def pyramid_map_parsing(self,input_layer):
    """
      Create a pyramid archiecture from thebinitial feature 

      Args:
          input_layer (_type_): Input to the layer

      Returns:
          _type_: pyramid prasing ouput
    """
    # Extraction of initial features map  
    feature_map = self.feature_map_extraction(input_layer)
    print('feature_map shape :', feature_map.shape)
    # performing average pooling on initial feature map to covert data into single tensor
    pyramid_red = GlobalAveragePooling2D(name='red_pool')(feature_map)
    pyramid_red = tf.keras.layers.Reshape((1,1,128))(pyramid_red)
    pyramid_red = Conv2D(filters=64,kernel_size=(1,1),name='red_1_by_1')(pyramid_red)
    # fine upsampling of data
    pyramid_red = UpSampling2D(size=256,interpolation='bilinear',name='red_upsampling')(pyramid_red)
    print('block_1 shape :', pyramid_red.shape)
    pyramid_yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(feature_map)
    pyramid_yellow = Conv2D(filters=64,kernel_size=(1,1),name='yellow_1_by_1')(pyramid_yellow)
    # normal upsampling of data
    pyramid_yellow = UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(pyramid_yellow)
    print('block_2 shape :', pyramid_yellow.shape)
    pyramid_blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(feature_map)
    pyramid_blue = Conv2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(pyramid_blue)
    # very coarse upsampling
    pyramid_blue = UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(pyramid_blue)
    print('block_3 shape :', pyramid_blue.shape)
    pyramid_green = AveragePooling2D(pool_size=(8,8),name='green_pool')(feature_map)
    pyramid_green = Conv2D(filters=64,kernel_size=(1,1),name='green_1_by_1')(pyramid_green)
    # coarse upsampling
    pyramid_green = UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(pyramid_green)
    print('block_4 shape :', pyramid_green.shape)
    pyramid_output=tf.keras.layers.concatenate([feature_map,pyramid_red,pyramid_yellow,pyramid_blue,pyramid_green])
    return pyramid_output
  
  def create_model(self,image_shape=(256,256,1),no_of_classes=4,disp=False):
    """
      Create PSP model by calling all the methods and building 

      Args:
          image_shape (tuple, optional): input image shape. Defaults to (256,256,1).
          no_of_classes (int, optional): Number of segmentation features. Defaults to 4.
          disp (bool, optional): Parameter to plot shapes and model archiecture. Defaults to False.

      Returns:
          _type_: PSP-net model
    """
    # building data pipeline by creating a input layer  
    pipe_line_inputs=tf.keras.Input(shape=image_shape,name='input')
    print("Building PSP-NET model \n")
    print('Input shape :', pipe_line_inputs.shape)
    # performing pyramid parsing of the data
    X = self.pyramid_map_parsing(pipe_line_inputs)
    print('pyramid map shape :', X.shape)
    # final convolution block (output block)
    X = Conv2D(filters=no_of_classes,kernel_size=3,padding='same',name='last_conv_3_by_3')(X)
    X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
    pipe_line_output = Activation('softmax',name='last_conv_relu')(X)
    print("output shape", pipe_line_output.shape, '\n')
    # PSP Net model is generated
    model = Model(inputs=pipe_line_inputs, outputs=pipe_line_output)
    print('Model built sucessfully \n')
    if disp:
      # plots network archiectutre  
      tf.keras.utils.plot_model(
              model, to_file='PSP_arch.png', show_shapes=True)
      model.summary()
    return model

class Thickness_estimation():
    """Thickness estimation build using different recurrent neural network
    """
    def __init__(self,number_of_classes=15,no_of_time_steps=200):
        """initial parameter required to build the network

        Args:
            number_of_classes (int, optional): Number of thickness classes. Defaults to 15.
            no_of_time_steps (int, optional): Lenght of the radiation phase. Defaults to 200.
        """
        self.number_of_classes=number_of_classes
        self.no_of_time_steps=no_of_time_steps
        pass
    
    def thickness_model(self,type='GRU',disp=False):
        """creates ML model for thickness estimation 

        Args:
            type (str, optional): Type of RNN. Defaults to 'GRU'.
            disp (bool, optional): parameter to plot model archiecture. Defaults to False.

        Returns:
            _type_: RNN model
        """
        if type=='GRU':
            thickness_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(1,self.no_of_time_steps))
                               ,tf.keras.layers.GRU(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.GRU(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.GRU(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.GRU(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_classes,activation='softmax')])
        elif type=='Bi-GRU':
            thickness_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(1,self.no_of_time_steps))
                               ,tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512,return_sequences=True,dropout=0.1))
                               ,tf.keras.layers.GRU(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.GRU(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_classes,activation='softmax')])
        elif type=='LSTM' :
            thickness_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(1,self.no_of_time_steps))
                               ,tf.keras.layers.LSTM(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.LSTM(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.LSTM(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.LSTM(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_classes,activation='softmax')])
        else :
            thickness_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(1,self.no_of_time_steps))
                               ,tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True,dropout=0.2))
                               ,tf.keras.layers.GRU(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.GRU(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_classes,activation='softmax')])
        if disp:
            tf.keras.utils.plot_model(
            thickness_model, to_file='thickness_estimation_arch.png', show_shapes=True)
            thickness_model.summary()
        return thickness_model

class Depth_estimation:

    def __init__(self,number_of_features=3,no_of_time_steps=200):
        """initial parameter required to build the network

        Args:
            number_of_features (int, optional): Number of segmentation features. Defaults to 3.
            no_of_time_steps (int, optional): Lenght of the radiation phase. Defaults to 200.
        """
        self.number_of_features=number_of_features
        self.no_of_time_steps=no_of_time_steps
        pass

    def depth_model(self,type='GRU',disp=False):
        """creates ML model for depth estimation 

        Args:
            type (str, optional): Type of RNN. Defaults to 'GRU'.
            disp (bool, optional): parameter to plot model archiecture. Defaults to False.

        Returns:
            _type_: RNN model
        """
        if type=='GRU':
            depth_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(self.no_of_time_steps,self.number_of_features))
                               ,tf.keras.layers.GRU(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.GRU(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.GRU(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.GRU(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_features,activation='softmax')])
        elif type=='Bi-GRU':
            depth_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(self.no_of_time_steps,self.number_of_features))
                               ,tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512,return_sequences=True,dropout=0.1))
                               ,tf.keras.layers.GRU(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.GRU(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_features,activation='softmax')])
        elif type=='LSTM' :
            depth_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(self.no_of_time_steps,self.number_of_features))
                               ,tf.keras.layers.LSTM(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.LSTM(512,return_sequences=True,dropout=0.2)
                               ,tf.keras.layers.LSTM(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.LSTM(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_features,activation='softmax')])
        else :
            depth_model=tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(self.no_of_time_steps,self.number_of_features))
                               ,tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True,dropout=0.2))
                               ,tf.keras.layers.GRU(256,return_sequences=True,dropout=0.1)
                               ,tf.keras.layers.GRU(128)
                               ,tf.keras.layers.Dense(256,activation='relu')
                               ,tf.keras.layers.Dense(self.number_of_features,activation='softmax')])
        if disp:
            tf.keras.utils.plot_model(
            depth_model, to_file='thickness_estimation_arch.png', show_shapes=True)
            depth_model.summary()
        return depth_model
        
        

if __name__ == '__main__':
    print(1)
#number_of_classes = 2
# WEIGHTS=r'D:\Thermal evaluation\LaDECO\thermography-evaluation\thermography-evaluation\pre_trained_vgg_weights.h5'
    input_shape = (256, 256, 1)
    filter_list = np.array([8,16, 32, 64, 128, 256,512])
    a = U_Net(number_of_classes, input_shape, filter_list)
    a.summary()
#model = tf.keras.models.load_model(r"W:\ml_thermal_imaging\LaDECO\LADECO\LADECO\models\u-net\U_NET_segementation_model.h5")
#model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model,show_shapes=True)
# seg_model=FCN8()
# model=seg_model.auto_encoder()
