#FOR MODIFYING IMAGES AND ARRAYS
from datetime import datetime
import os,cv2
#from cv2 import getRotationMatrix2D, warpAffine,getAffineTransform,resize,imread,BORDER_REFLECT
import numpy as np
#KERAS IMPORTS
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, concatenate
from keras.layers.core import Reshape, Activation, Dropout
from keras.preprocessing.image import *
from keras.optimizers import SGD
#UTILITY GLOBAL VARIABLES
input_dim = [512,928]  
input_dim_tuple = (input_dim[0],input_dim[1])
num_class = 6
C=4
index = [0, 1020,1377  ,240, 735, 2380]





#================================================MODEL_ARCHITECTURE============================================================

# RGB MODALITY BRANCH OF CNN
inputs_rgb = Input(shape=(input_dim[0],input_dim[1],3))
vgg_model_rgb = VGG16(weights='imagenet', include_top = False,modality_num=0)
conv_model_rgb = vgg_model_rgb(inputs_rgb)
conv_model_rgb = Conv2D(64, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
conv_model_rgb = Conv2D(128, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
dropout_rgb = Dropout(0.4)(conv_model_rgb)

# NIR MODALITY BRANCH OF CNN
inputs_nir = Input(shape=(input_dim[0],input_dim[1],3))
vgg_model_nir = VGG16(weights='imagenet', include_top= False,modality_num=1)
conv_model_nir = vgg_model_nir(inputs_nir)
conv_model_nir = Conv2D(64, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_nir)
conv_model_nir = Conv2D(128, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_nir)
dropout_nir = Dropout(0.4)(conv_model_nir)

# CONACTENATE the ends of RGB & NIR 
merge_rgb_nir = concatenate([conv_model_nir, conv_model_rgb