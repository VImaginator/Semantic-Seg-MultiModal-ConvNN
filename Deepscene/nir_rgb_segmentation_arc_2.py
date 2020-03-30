#FOR MODIFYING IMAGES AND ARRAYS
import os, cv2
import numpy as np

#KERAS IMPORTS
import keras
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, core
from keras.preprocessing.image import *

#UTILITY GLOBAL VARIABLES
input_dim = [256, 480]  
num_class = 6
C = 10
index = [2380, 1020,  969,  240, 2775,    0]

#HELPER FUNCTION OF SEGMENT_DATA_GENERATOR
# comprises of path and extension of images in a directory
class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext
        

#RESIZES 3D IMAGES(image)(EX: RGB) TO DESIRED SIZE(crop_size) 
def fix_size(image, crop_size):
    cropy, cropx = crop_size
    height, width = image.shape[:-1]
    
    #adjusting height of the image 
    cy = cropy - height
    if cy > 0:
        if cy % 2 == 0:
            image = np.vstack((np.zeros((cy/2,width,3)) , image , np.zeros((cy/2,width,3))))
        else:
            image = np.vstack((np.zeros((cy/2,width,3)) , image , np.zeros((cy/2 +1,width,3))))
    if cy < 0:
        if cy % 2 == 0:
            image = np.delete(image, range(-1*cy/2), axis = 0)
            image = np.delete(image, range(height + cy,height +  cy/2), axis = 0)
        else:
            image = np.delete(image, range(-1*cy/2), axis =0)
            image = np.delete(image, range(height + cy, height + cy/2 + 1), axis=0)
    
    #adjusting width of the image
    height, width = image.shape[:-1]
    cx = cropx - width
    if cx > 0:
        if cx % 2 == 0:
            image = np.hstack((np.zeros((height,cx/2,3)) , image , np.zeros((height,cx/2,3))))
        else:
            image = np.hstack((np.zeros((height,cx/2,3)) , image , np.zeros((height,cx/2 + 1,3))))
    if cx < 0:
        if cx % 2 == 0:
            image = np.delete(image, range(-1*cx/2), axis = 1)
            image = np.delete(image, range(width + cx,width +  cx/2), axis = 1)
        else:
            image = np.delete(image, range(-1*cx/2), axis =1)
            image = np.delete(image, range(width + cx, width + cx/2 + 1), axis=1)
    return image


#CONVERTING Ground Truth IMAGES(image) TO A ARRAY OF PIXELWISE ONE-HOT VECTORS(of dimension 'no_class')
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
class aug_state:
    def __init__(self,flip_axis_index=0,rotation_range=360,height_range=0.2,width_range=0.2,shear_intensity=1,color_intensity=40,zoom_range=(1.2,1.2)):
         self.flip_axis_index=flip_axis_index
         self.rotation_range=rotation_range
         self.height_range=height_range
         self.width_range=width_range
         self.shear_intensity=shear_intensity
         self.color_intensity=color_intensity
         self.zoom_range=zoom_range


def data_augmentor(x,state,row_axis=0,col_axis=1,channel_axis=-1,
    bool_flip_axis=True,
    bool_random_rotation=True,
    bool_random_shift=True,
    bool_random_shear=True,
    bool_random_channel_shift=True,
    bool_random_zoom=True):
    if bool_flip_axis:
        flip_axis(x, state.flip_axis_index)

    if bool_random_rotation:
        random_rotation(x, state.rotation_range, row_axis, col_axis, channel_axis)

    if bool_random_shift:
        random_shift(x, state.width_range, state.height_range, row_axis, col_axis, channel_axis)

    if bool_random_shear:
        random_shear(x, state.shear_intensity, row_axis, col_axis, channel_axis)

    if bool_random_channel_shift:
        random_channel_shift(x, state.color_intensity, channel_axis)

    if bool_random_zoom:
        random_zoom(x, state.zoom_range, row_axis, col_axis, channel_axis)

    return x



#=====================================================================================================================
#DATAGENERATOR FOR MULTIMODAL SEMANTIC SEGMENTATION
def Segment_datagen(state_aug,file_path, rgb_args, nir_args, label_args, batch_size, input_size):
    # Create MEMORY enough for one batch of input(s) + augmentation & labels + augmentation
    data = np.zeros((2,batch_size*2,input_size[0],input_size[1],3))
    labels = np.zeros((batch_size*2,input_size[0]*input_size[1],6))
    # Read the file names
    files = open(file_path)
    names = files.readlines()
    files.close()
    # Enter the indefinite loop of generator
    while True:
        for i in range(batch_size):
            index_of_random_sample = np.random.choice(len(names))
   