# -*- coding: utf-8 -*-
## This file is used to plot learning curve of the training process of different models
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning on thermographic videos)

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(scalars, weight) :  # Weight between 0 and 1
    """taken from open stacks 
    weight average operation to remove flutations
    Args:
        scalars (_type_): input data 
        weight (_type_): weigth factor 

    Returns:
        _type_: smoothed data
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

file_path=r"D:\LaDECO\LaDECO\trained_models\GRU\thickness_di_gru.pkl"
pickle_file = open(file_path, "rb")
objects = []
while True:
    try:
        objects.append(pickle.load(pickle_file))
    except EOFError:
        break
pickle_file.close()


learning_history = pd.read_pickle(file_path) 
print(learning_history.keys())

plt.figure(figsize=(8,4))
plt.plot(learning_history['loss'],color='tab:blue',label='loss',alpha=0.5)
plt.plot(learning_history['val_loss'],color='tab:orange',label='val loss',alpha=0.5)
plt.plot(smooth(learning_history['loss'],0.9),color='tab:blue',linewidth=2,label='loss smooth')
plt.plot(smooth(learning_history['val_loss'],0.9),color='tab:orange',linewidth=2,label='val loss smooth')
#plt.scatter(0,learning_history['loss'][0],color='tab:red')
#plt.text(0.5,learning_history['loss'][0],np.round(learning_history['loss'][0],1),color='tab:red')
#plt.scatter(200,learning_history['loss'][-1],color='tab:green')
#plt.text(199,learning_history['loss'][-1]+0.06,np.round(learning_history['loss'][-1],1),color='tab:green')
#plt.scatter(200,learning_history['val_loss'][-1],color='tab:green')
#plt.text(199,learning_history['val_loss'][-1]+0.06,np.round(learning_history['val_loss'][-1],1),color='tab:green')
plt.grid(linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Bi-GRU thickness estimation loss learning curve')
plt.legend()
plt.savefig(r"D:\STUDY_MATERIAL\document\thicknesss_final_loss.png",dpi=600,bbox_inches='tight',transparent=True)
#plt.show()


plt.figure(figsize=(8,4))
plt.plot(learning_history['accuracy'],color='tab:blue',alpha=0.5,label='accuracy')
plt.plot(learning_history['val_accuracy'],color='tab:orange',alpha=0.5,label='val accuracy')
plt.plot(smooth(learning_history['accuracy'],0.9),color='tab:blue',linewidth=2,label='loss smooth')
plt.plot(smooth(learning_history['val_accuracy'],0.9),color='tab:orange',linewidth=2,label='val loss smooth')
#plt.scatter(0,learning_history['accuracy'][0],color='tab:red')
#plt.text(0.5,learning_history['accuracy'][0],np.round(learning_history['accuracy'][0],1),color='tab:red')
#plt.scatter(200,learning_history['accuracy'][-1],color='tab:green')
#plt.text(199,learning_history['accuracy'][-1]-0.06,np.round(learning_history['accuracy'][-1],1),color='tab:green')
#plt.scatter(200,learning_history['val_accuracy'][-1],color='tab:green')
#plt.text(199,learning_history['val_accuracy'][-1]+0.06,np.round(learning_history['val_accuracy'][-1],1),color='tab:green')
plt.grid(linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Bi-GRU thickness estimation accuracy learning curve')
plt.legend()
plt.savefig(r"D:\STUDY_MATERIAL\document\thickness_final_acc.png",dpi=600,bbox_inches='tight',transparent=True)
plt.show()
"""
file_path1=r"D:\LaDECO\LaDECO\trained_models\GRU\thickness_di_gru.pkl"
file_path2=r'D:\LaDECO\LaDECO\trained_models\GRU\thickness_di_gru_std.pkl'

learning_history_xception = pd.read_pickle(file_path1) 
learning_history_inception = pd.read_pickle(file_path2) 
plt.figure(figsize=(8,4))
plt.plot(learning_history_xception['loss'],alpha=0.2,color='tab:blue',linewidth=1)
plt.plot(smooth(learning_history_xception['loss'],0.9),color='tab:blue',linewidth=2,label='contrast function')
#plt.text(199,learning_history_xception['loss'][-1]+0.06,np.round(learning_history_xception['loss'][-1],1),color='tab:blue')
#plt.scatter(200,learning_history_xception['loss'][-1],color='tab:blue')
plt.plot(learning_history_inception['loss'],color='tab:orange',alpha=0.2,linewidth=1)
plt.plot(smooth(learning_history_inception['loss'],0.9),color='tab:orange',linewidth=2,label='Normalization')
#plt.text(199,learning_history_inception['loss'][-1]-0.3,np.round(learning_history_inception['loss'][-1],1),color='tab:orange')
#plt.scatter(200,learning_history_inception['loss'][-1],color='tab:orange')

#plt.text(199,learning_history_resnet['loss'][-1]+0.3,np.round(learning_history_resnet['loss'][-1],1),color='tab:green')
#plt.scatter(200,learning_history_resnet['loss'][-1],color='tab:green')
plt.grid(linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,2)
plt.title('RNN loss learning curve')
plt.legend()
plt.savefig(r"D:\STUDY_MATERIAL\document\RNN_contrast_loss.png",dpi=600,bbox_inches='tight',transparent=True)


learning_history_xception = pd.read_pickle(file_path1) 
learning_history_inception = pd.read_pickle(file_path2) 
#learning_history_resnet=pd.read_pickle(file_path3) 
plt.figure(figsize=(8,4))
plt.plot(learning_history_xception['accuracy'],alpha=0.2,color='tab:blue',linewidth=1)
plt.plot(smooth(learning_history_xception['accuracy'],0.9),color='tab:blue',linewidth=2,label='contrast function')
#plt.text(199,learning_history_xception['loss'][-1]+0.06,np.round(learning_history_xception['loss'][-1],1),color='tab:blue')
#plt.scatter(200,learning_history_xception['loss'][-1],color='tab:blue')
plt.plot(learning_history_inception['accuracy'],color='tab:orange',alpha=0.2,linewidth=1)
plt.plot(smooth(learning_history_inception['accuracy'],0.9),color='tab:orange',linewidth=2,label='Normalization')
#plt.text(199,learning_history_resnet['loss'][-1]+0.3,np.round(learning_history_resnet['loss'][-1],1),color='tab:green')
#plt.scatter(200,learning_history_resnet['loss'][-1],color='tab:green')
plt.grid(linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('RNN accuracy learning curve')
plt.legend()
plt.savefig(r"D:\STUDY_MATERIAL\document\RNN_contrast_accuracy.png",dpi=600,bbox_inches='tight',transparent=True)
plt.show()
"""
file_path1=r"D:\LaDECO\LaDECO\trained_models\GRU\thickness_di_gru.pkl"
file_path2=r'D:\LaDECO\LaDECO\trained_models\GRU\thickness_di_gru_03_dp.pkl'
file_path3=r'D:\LaDECO\LaDECO\trained_models\GRU\thickness_di_gru_05_dp.pkl'
#file_path4=r"D:\LaDECO\LaDECO\trained_models\LSTM\thickness_bi_lstm.pkl"
learning_history_xception = pd.read_pickle(file_path1) 
learning_history_inception = pd.read_pickle(file_path2) 
learning_history_resnet=pd.read_pickle(file_path3) 
#learning_history_resnet1=pd.read_pickle(file_path4) 
plt.figure(figsize=(8,4))
plt.plot(learning_history_xception['loss'],alpha=0.2,color='tab:blue',linewidth=1)
plt.plot(smooth(learning_history_xception['loss'],0.9),color='tab:blue',linewidth=2,label='0.2')
#plt.text(199,learning_history_xception['loss'][-1]+0.06,np.round(learning_history_xception['loss'][-1],1),color='tab:blue')
#plt.scatter(200,learning_history_xception['loss'][-1],color='tab:blue')
plt.plot(learning_history_inception['loss'],color='tab:orange',alpha=0.2,linewidth=1)
plt.plot(smooth(learning_history_inception['loss'],0.9),color='tab:orange',linewidth=2,label='0.3')
#plt.text(199,learning_history_inception['loss'][-1]-0.3,np.round(learning_history_inception['loss'][-1],1),color='tab:orange')
#plt.scatter(200,learning_history_inception['loss'][-1],color='tab:orange')
plt.plot(learning_history_resnet['loss'],alpha=0.2,color='tab:green',linewidth=1)
plt.plot(smooth(learning_history_resnet['loss'],0.9),color='tab:green',linewidth=2,label='0.5')
#plt.plot(learning_history_resnet1['loss'],alpha=0.2,color='tab:olive',linewidth=1)
#plt.plot(smooth(learning_history_resnet1['loss'],0.9),color='tab:olive',linewidth=2,label='Bi-LSTM')
#plt.text(199,learning_history_resnet['loss'][-1]+0.3,np.round(learning_history_resnet['loss'][-1],1),color='tab:green')
#plt.scatter(200,learning_history_resnet['loss'][-1],color='tab:green')
plt.grid(linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,2)
plt.title('RNN loss learning curve drop-out parameter')
plt.legend()
plt.savefig(r"D:\STUDY_MATERIAL\document\RNN_dp_loss.png",dpi=600,bbox_inches='tight',transparent=True)


learning_history_xception = pd.read_pickle(file_path1) 
learning_history_inception = pd.read_pickle(file_path2) 
learning_history_resnet=pd.read_pickle(file_path3) 
plt.figure(figsize=(8,4))
plt.plot(learning_history_xception['accuracy'],alpha=0.2,color='tab:blue',linewidth=1)
plt.plot(smooth(learning_history_xception['accuracy'],0.9),color='tab:blue',linewidth=2,label='0.2')
#plt.text(199,learning_history_xception['loss'][-1]+0.06,np.round(learning_history_xception['loss'][-1],1),color='tab:blue')
#plt.scatter(200,learning_history_xception['loss'][-1],color='tab:blue')
plt.plot(learning_history_inception['accuracy'],color='tab:orange',alpha=0.2,linewidth=1)
plt.plot(smooth(learning_history_inception['accuracy'],0.9),color='tab:orange',linewidth=2,label='0.3')
#plt.text(199,learning_history_inception['loss'][-1]-0.3,np.round(learning_history_inception['loss'][-1],1),color='tab:orange')
#plt.scatter(200,learning_history_inception['loss'][-1],color='tab:orange')
plt.plot(learning_history_resnet['accuracy'],alpha=0.2,color='tab:green',linewidth=1)
plt.plot(smooth(learning_history_resnet['accuracy'],0.9),color='tab:green',linewidth=2,label='0.5')
#plt.plot(learning_history_resnet1['accuracy'],alpha=0.2,color='tab:olive',linewidth=1)
#plt.plot(smooth(learning_history_resnet1['accuracy'],0.9),color='tab:olive',linewidth=2,label='Bi-LSTM')

plt.grid(linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('RNN accuracy learning curve drop-out parameter')
plt.legend()
plt.savefig(r"D:\STUDY_MATERIAL\document\RNN_dp_accuracy.png",dpi=600,bbox_inches='tight',transparent=True)
plt.show()
