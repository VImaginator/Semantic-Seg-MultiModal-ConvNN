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
num_