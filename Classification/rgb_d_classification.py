from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import convolutional, pooling, core, Input, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger

img_width, img_height = 256, 256
train_data_dir = "../Videos/rgb-dataset"
validation_data_dir = "../Videos/rgb-dataset"
train_data_dir_nir = "../Videos/d-dataset"
