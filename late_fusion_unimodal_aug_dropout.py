
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
input_dim = (512,928)
dim_tup = (928,512)
num_class = 6
C = 4
index = [0, 1020, 1377  , 240, 735, 2380]
#HELPER FUNCTION OF SEGMENT_DATA_GENERATOR
# comprises of path and extension of images in a directory
class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext
        
#CONVERTING MASKED IMAGES(image) TO A ARRAY OF PIXELWISE ONE-HOT VECTORS(of dimension 'no_class')
def fix_label(image, no_class):
    width , height, depth = image.shape
    #generating hashes for each pixel (index array above has the hash values for each class)
    image = np.dot(image.reshape(width*height,depth)[:,],[1,4,9])
    #converting hashes to indices of classes
    for i in range(no_class):
        image[image == index[i]] = i
    #converting each index into one-hot vector of dim of classes(no_class)
    image = (np.arange(no_class) == image[...,None])*1
    return image


#====================================================data==augmentation==============================================================
'''class aug_state:
    def __init__(self,flip_axis_index=0,zoom_range=(1.2,1.2)):
         self.flip_axis_index=flip_axis_index
         self.zoom_range=zoom_range
         
def data_augmentor(x,state,row_axis=1,col_axis=0,channel_axis=-1):
    #dt = datetime.now()
    #(int(str(dt).split('.')[1])%100)
    t = np.random.randint(4,size=2)
    temp =[0,0,0,0,0]
    temp[t[0]] = 1
    temp[t[1]] = 1
    #print temp
    if temp[0]:
        x = flip_axis(x, state.flip_axis_index)
  
    if temp[1]:
        M = cv2.getRotationMatrix2D((x.shape[1]/2,x.shape[0]/2),np.random.randint(360),1)   #last arg