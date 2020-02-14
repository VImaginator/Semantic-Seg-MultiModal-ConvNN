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
validation_data_dir_nir = "../Videos/d-dataset"
nb_train_samples = 207920
nb_validation_samples = 2079
batch_size = 10
epochs = 4
num_classes = 51


inp = Input(shape = (img_width , img_height, 3))
conv_layer1 = convolutional.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu')(inp)
conv_layer2 = convolutional.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu')(conv_layer1)
pool_layer1 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer2)
conv_layer3 = convolutional.Conv2D(16, (3,3), strides=(1, 1), padding='same', activation='relu')(pool_layer1)
pool_layer2 = pooli