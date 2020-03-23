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
    for i