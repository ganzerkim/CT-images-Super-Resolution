# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:01:45 2019

@author: MIT-DGMIF
"""

import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2 as cv
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, Dense, UpSampling2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from skimage.transform import pyramid_reduce 

import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img#,save_img
from Subpixel import Subpixel



base_path = 'D:\\microscopy\\'
x_train = np.load(base_path + 'dataset\\x_train.npy')
y_train = np.load(base_path + 'dataset\\y_train.npy')
x_val = np.load(base_path + 'dataset\\x_val.npy')
y_val = np.load(base_path + 'dataset\\y_val.npy')


"""
base_path = 'D:\\H&E_dataset\\'

x_train = np.load(base_path + 'dataset\\x_train_220025(40).npy')
y_train = np.load(base_path + 'dataset\\y_train_220025(100).npy')
x_val = np.load(base_path + 'dataset\\x_val_220025(40).npy')
y_val = np.load(base_path + 'dataset\\y_val_220025(100).npy')
"""


#x_train = x_train[:, :, :, np.newaxis]
#y_train = y_train[:, :, :, np.newaxis]
#x_val = x_val[:, :, :, np.newaxis]
#y_val = y_val[:, :, :, np.newaxis]

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)



####################################################3
def MG_Batch(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def MG_Conv(x, filters, size, strides=(1,1), padding='same', activation = True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = MG_Batch(x)
    return x

def MG_residual(blockInput, num_filters = 64, l = 4):
    x = MG_Batch(blockInput)
    x = MG_Conv(blockInput, num_filters, (3, 3), activation = False)   
    x2 = Add()([x, blockInput])
    
    if l == 1:
        return x2 
    else:
        return MG_residual(x2, num_filters = num_filters, l = l - 1)



def MG_dense(blockInput, num_filters = 64, activation = True):
    
    x1_1 = MG_Batch(blockInput)
    x1_2 = MG_Conv(x1_1, num_filters, (3, 3), activation = False)
    
    d1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x1_2)
    d1_2 = MG_Conv(d1_1, num_filters, (1, 1), activation = activation)
    d1_3 = MG_Conv(d1_2, int(num_filters/4), (3, 3), activation = activation)
    
    c1 = concatenate([d1_3, blockInput])
    
    x2_1 = MG_Batch(c1)
    x2_2 = MG_Conv(x2_1, num_filters, (3, 3), activation = False)
    
    d2_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x2_2)
    d2_2 = MG_Conv(d2_1, num_filters, (1, 1), activation = activation)
    d2_3 = MG_Conv(d2_2, int(num_filters/4), (3, 3), activation = activation)
    
    c2 = concatenate([d2_3, d1_3, blockInput])
    
    x3_1 = MG_Batch(c2)
    x3_2 = MG_Conv(x3_1, num_filters, (3, 3), activation = False)
    
    d3_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x3_2)
    d3_2 = MG_Conv(d3_1, num_filters, (1, 1), activation = activation)
    d3_3 = MG_Conv(d3_2, int(num_filters/4), (3, 3), activation = activation)
    
    c3 = concatenate([d3_3, d2_3, d1_3, blockInput])
    
    x4_1 = MG_Batch(c3)
    x4_2 = MG_Conv(x4_1, num_filters, (3, 3), activation = False)
    
    d4_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x4_2)
    d4_2 = MG_Conv(d4_1, num_filters, (1, 1), activation = activation)
    d4_3 = MG_Conv(d4_2, int(num_filters/4), (3, 3), activation = activation)
    
    c4 = concatenate([d4_3, d3_3, d2_3, d1_3, blockInput])
    
    x = x1_2 = MG_Conv(c4, num_filters, (3, 3), activation = False)
    
    return x
    
    
def MG_recursive(blockInput, num_filters=16, batch_activate = False, recur_time = 2):
    x = MG_Conv(blockInput, num_filters, (3,3), activation = False)
    x = MG_Conv(x, num_filters, (3,3), activation = False)
    x = Add()([x, blockInput])
    if recur_time == 0:        
        return Add()([x, MG_residual(x, num_filters, batch_activate)])
    
    else:
        return Add()([x, MG_recursive(x, num_filters, batch_activate, recur_time = recur_time - 1)])


def MG_recur(blockInput, num_filters=16, batch_activate = False, recur_time = 2):
    x1 = MG_Conv(blockInput, num_filters, (3,3), batch_activation = batch_activate)
    x1 = MG_Conv(x1, num_filters, (3,3), activation = batch_activate)
    #x2 = MG_Conv(x2, num_filters, (3,3), activation = False)
    x = Add()([x1, blockInput])
    
    if recur_time == 0:        
        return x
    
    else:
        return Add()([x, MG_recur(x, num_filters, batch_activate, recur_time = recur_time - 1)])

upscale_factor = 2
L = 2.5

inputs = Input(shape=(40, 40, 3))

net_input_1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
net_input_3 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net_input_5 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
net_input_9 = Conv2D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu')(inputs)
net_input_11 = Conv2D(filters=32, kernel_size=11, strides=1, padding='same', activation='relu')(inputs)

net1 = MG_dense(net_input_1, num_filters = 32, activation = True)
r1 = Add()([net_input_1, net1])
g1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(r1)

net2 = MG_dense(net_input_3, num_filters = 32, activation = True)
r2 = Add()([net_input_3, net2])
g2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(r2)

net3 = MG_dense(net_input_5, num_filters = 32, activation = True)
r3 = Add()([net_input_5, net3])
g3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(r3)

net4 = MG_dense(net_input_9, num_filters = 32, activation = True)
r4 = Add()([net_input_9, net3])
g4 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(r4)

net5 = MG_dense(net_input_11, num_filters = 32, activation = True)
r5 = Add()([net_input_11, net3])
g5 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(r5)

input_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
r6 = Add()([g1, g2, g3, g4, g5, input_2])

sub_net_1 = Conv2D(filters=75, kernel_size=3, strides=1, padding='same', activation='relu')(r6)
sub_net_2 = Conv2D(filters=75, kernel_size=9, strides=1, padding='same', activation='relu')(r6)
sub_net_1 = Subpixel(filters=75, kernel_size=3, r=int(L*upscale_factor), padding='same')(sub_net_1)
sub_net_2 = Subpixel(filters=75, kernel_size=9, r=int(L*upscale_factor), padding='same')(sub_net_2)

sub_net = Add()([sub_net_1, sub_net_2])
sub_net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(sub_net)
sub_net = Conv2D(filters=3, kernel_size=3, strides=2, padding='same', activation='relu')(sub_net)
sub_net = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='relu')(sub_net)

outputs = Activation('relu')(sub_net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse',  metrics=['acc', 'mse'])

model.summary()

from keras.utils import plot_model
plot_model(model, to_file = 'RDB_MG_modified_200123.png')


"""
#%%
#image datagenerator


# 랜덤시드 고정시키기
np.random.seed(5)

# 랜덤시드 고정시키기
np.random.seed(3)

from keras.preprocessing.image import ImageDataGenerator

# 데이터셋 불러오기
train_datagen = ImageDataGenerator(rescale=1, 
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

data_flow = train_datagen.flow(x_train, y_train, batch_size = 16)

history = model.fit_generator(data_flow, epochs=500, steps_per_epoch=1500,
                                  verbose=1, validation_data=(x_val, y_val))
 
performance_test = model.evaluate(x_val, y_val, batch_size=16, verbose=2)

print('\nTest Result ->', performance_test)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
 
def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
 
plot_acc(history)
plt.show()
plot_loss(history)
plt.show()


model.save('D:\\microscopy\\training_RDB_h&e_191210.h5')
print("Saved model to disk")

#Evaluation
model.save_weights('D:\\microscopy\\weights_training_RDB_h&e_191210.h5')
"""
#%%



#Train

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

#callbacks = [early_stopping] 옵션

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=8, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2, mode='auto', min_lr=1e-05)
])

model.save('D:\\H&E_dataset\\training_DENSE_RES_modified_MG_200123.h5')
print("Saved model to disk")

#Evaluation
model.save_weights('D:\\H&E_dataset\\weight_training_DENSE_RES_modified_MG_200123.h5')


load_model('C:\\Users\\MG\\Desktop\\H&E New dataset\\SuperResolutionData\\CT_imageset\\20200129\\training_DENSE_RES_modified_MG_200123.h5')

model.load_weights('C:\\Users\\MG\\Desktop\\H&E New dataset\\SuperResolutionData\\CT_imageset\\20200129\\weight_training_DENSE_RES_modified_MG_200123.h5')

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')


test_img_path = 'C:\\Users\\MG\\Desktop\\pre'
test_img = [cv.imread(test_img_path + '\\' + s) for s in os.listdir(test_img_path)]

#x_val = x_val[np.newaxis, :, :, :]
test_imgset = np.array(test_img)
test_imgset_norm = cv.normalize(test_imgset.astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
preds = model.predict(test_imgset_norm)



indx = 1

x_val_re = cv.normalize(test_imgset_norm[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
#y_val_re = cv.normalize(y_val[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
preds_re = cv.normalize(preds[indx].astype(np.float64), None, 0, 295, cv.NORM_MINMAX)

preds_re = preds_re.astype(np.uint8)
hsv_img = cv.cvtColor(preds_re, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv_img)

clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
clahe_v = clahe.apply(v)
results = cv.merge((h, s, clahe_v))
results = cv.cvtColor(results, cv.COLOR_HSV2BGR)

cv.imwrite(test_img_path + '\\x_val_3.png', x_val_re)
cv.imwrite(test_img_path + '\\prd_MG_modified_0_295.png', preds_re)



indx = 2000

x_val_re = cv.normalize(x_val[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
y_val_re = cv.normalize(y_val[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
preds_re = cv.normalize(preds[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)

iii = 0
preds_tmp = []
for iii in range(len(preds)):
    preds_down = cv.resize(preds[iii], dsize = (40, 40), interpolation = cv.INTER_AREA)
    preds_tmp.append(preds_down)

preds_tmp = np.array(preds_tmp)
preds_twice = model.predict(preds_tmp)
preds_twice_re = cv.normalize(preds_twice[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)

ii = 0
preds_tmp_3 = []
for ii in range(len(preds_tmp)):
    preds_down_3 = cv.resize(preds_twice[ii], dsize = (40, 40), interpolation = cv.INTER_AREA)
    preds_tmp_3.append(preds_down_3)

preds_tmp_3 = np.array(preds_tmp_3)
preds_third = model.predict(preds_tmp_3)
preds_third_re = cv.normalize(preds_third[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)




f, axs = plt.subplots(3,2,figsize=(10,10))
plt.subplot(331),plt.imshow(x_val_re[:,:,0])
plt.title('low resolution'), plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(y_val_re[:,:,0])
plt.title('Ground Truth Image'), plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(preds_re[:,:,0])
plt.title('Once output'), plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.imshow(preds_twice_re[:, :, 0])
plt.title('Twice output'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(preds_third_re[:, :, 0])
plt.title('Third output'), plt.xticks([]), plt.yticks([])
plt.show()


#------------이미지 구성----------------#
#------x_val----------------------------#
x_val_R = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val_R.png', x_val_re[:,:,0])
x_val_G = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val_G.png', x_val_re[:,:,1])
x_val_B = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val_B.png', x_val_re[:,:,2])

x_r = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val_R.png', cv.IMREAD_GRAYSCALE)
x_g = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val_G.png', cv.IMREAD_GRAYSCALE)
x_b = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val_B.png', cv.IMREAD_GRAYSCALE)

img_x = cv.merge([x_r, x_g, x_b])
plt.imshow(img_x)
plt.show()

cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val.png', x_val_re)

#---------------------------------------------#
y_val_R = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val_R.png', y_val_re[:,:,0])
y_val_G = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val_G.png', y_val_re[:,:,1])
y_val_B = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val_B.png', y_val_re[:,:,2])

y_r = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val_R.png', cv.IMREAD_GRAYSCALE)
y_g = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val_G.png', cv.IMREAD_GRAYSCALE)
y_b = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val_B.png', cv.IMREAD_GRAYSCALE)

img_y = cv.merge([y_r, y_g, y_b])
plt.imshow(img_y)
plt.show()
cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val.png', y_val_re)

#----------------------------------------------------#


preds_R = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\preds_R.png', preds_re[:,:,0])
preds_G = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\preds_G.png', preds_re[:,:,1])
preds_B = cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\preds_B.png', preds_re[:,:,2])

preds_r = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\preds_R.png', cv.IMREAD_GRAYSCALE)
preds_g = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\preds_G.png', cv.IMREAD_GRAYSCALE)
preds_b = cv.imread('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\preds_B.png', cv.IMREAD_GRAYSCALE)

img_preds = cv.merge([preds_r, preds_g, preds_b])
plt.imshow(img_preds)
plt.show()
cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\preds\\prds.png', preds_re)


#--------------------------------------------------------------#

preds_2_R = cv.imwrite('D:\\microscopy\\img_save\\preds_2\\preds_2_R.png', preds_twice_re[:,:,0])
preds_2_G = cv.imwrite('D:\\microscopy\\img_save\\preds_2\\preds_2_G.png', preds_twice_re[:,:,1])
preds_2_B = cv.imwrite('D:\\microscopy\\img_save\\preds_2\\preds_2_B.png', preds_twice_re[:,:,2])

preds_2_r = cv.imread('D:\\microscopy\\img_save\\preds_2\\preds_2_R.png', cv.IMREAD_GRAYSCALE)
preds_2_g = cv.imread('D:\\microscopy\\img_save\\preds_2\\preds_2_G.png', cv.IMREAD_GRAYSCALE)
preds_2_b = cv.imread('D:\\microscopy\\img_save\\preds_2\\preds_2_B.png', cv.IMREAD_GRAYSCALE)

img_preds_2 = cv.merge([preds_2_r, preds_2_g, preds_2_b])
plt.imshow(img_preds_2)
plt.show()
cv.imwrite('D:\\microscopy\\img_save\\preds_2\\preds_2.png', img_preds_2)

#--------------------------------------------------------------------#


preds_3_R = cv.imwrite('D:\\microscopy\\img_save\\preds_3\\preds_3_R.png', preds_third_re[:,:,0])
preds_3_G = cv.imwrite('D:\\microscopy\\img_save\\preds_3\\preds_3_G.png', preds_third_re[:,:,1])
preds_3_B = cv.imwrite('D:\\microscopy\\img_save\\preds_3\\preds_3_B.png', preds_third_re[:,:,2])

preds_3_r = cv.imread('D:\\microscopy\\img_save\\preds_3\\preds_3_R.png', cv.IMREAD_GRAYSCALE)
preds_3_g = cv.imread('D:\\microscopy\\img_save\\preds_3\\preds_3_G.png', cv.IMREAD_GRAYSCALE)
preds_3_b = cv.imread('D:\\microscopy\\img_save\\preds_3\\preds_3_B.png', cv.IMREAD_GRAYSCALE)

img_preds_3 = cv.merge([preds_3_r, preds_3_g, preds_3_b])
plt.imshow(img_preds_3)
plt.show()
cv.imwrite('D:\\microscopy\\img_save\\preds_3\\preds_3.png', img_preds_3)

