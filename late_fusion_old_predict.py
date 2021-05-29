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
merge_rgb_nir = concatenate([conv_model_nir, conv_model_rgb], axis=-1)

# DECONVOLUTION Layers
deconv_last = Conv2DTranspose(num_class, (64,64), strides=(32, 32), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal') (merge_rgb_nir)

#VECTORIZING OUTPUT
out_reshape = Reshape((input_dim[0]*input_dim[1],num_class))(deconv_last)
out = Activation('softmax')(out_reshape)

# MODAL [INPUTS , OUTPUTS]
model = Model(inputs=[inputs_rgb, inputs_nir], outputs=[out])

print 'compiling'
model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("late_fusion_working_noaug.hdf5",by_name=True)
model.summary()

def fix_label(image, no_class):
        width , height, depth = image.shape
        #generating hashes for each pixel (index array above has the hash values for each class)
        image = np.dot(image.reshape(width*height,depth)[:,],[1,4,9])
  