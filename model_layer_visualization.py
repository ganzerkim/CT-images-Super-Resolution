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

import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img#,save_img
from Subpixel import Subpixel



base_path = 'D:\\Data set\\SRdataset\\'
x_train = np.load(base_path + 'dataset\\x_train.npy')
y_train = np.load(base_path + 'dataset\\y_train.npy')
x_val = np.load(base_path + 'dataset\\x_val.npy')
y_val = np.load(base_path + 'dataset\\y_val.npy')

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

"""
def MG_residual(blockInput, num_filters=16, batch_activate = False):
    x = MG_Batch(blockInput)
    x = MG_Conv(x, num_filters, (3,3), activation = True)
    x = MG_Conv(x, num_filters, (3,3), activation = True)
    x = Add()([x, blockInput])
    if batch_activate == True:
        x = MG_Batch(x)
    return x
"""

def MG_residual(blockInput, num_filters = 64, l = 4):
    x = MG_Conv(blockInput, num_filters, (3, 3), activation = True)
    x = MG_Conv(x, num_filters, (3, 3), activation = True)
    x = MG_Batch(blockInput)
    x2 = Add()([x, blockInput])
    
    if l == 1:
        return x2 
    else:
        return MG_residual(x2, num_filters = num_filters, l = l - 1)



def MG_dense(blockInput, num_filters = 64, activation = True):
    x1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(blockInput)
    x1 = MG_Conv(x1, 32, (1, 1), activation = activation)
    x1 = MG_Conv(x1, 8, (3, 3), activation = activation)
    
    x2 = concatenate([x1, blockInput])
    x2 = Conv2D(filters=40, kernel_size=3, strides=1, padding='same', activation='relu')(x2)
    x2 = MG_Conv(x1, 32, (1, 1), activation = activation)
    x2 = MG_Conv(x1, 8, (3, 3), activation = activation)
    
    x3 = concatenate([x2, x1, blockInput])
    x3 = Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation='relu')(x3)
    x3 = MG_Conv(x1, 32, (1, 1), activation = activation)
    x3 = MG_Conv(x1, 8, (3, 3), activation = activation)
    
    x4 = concatenate([x3, x2, x1, blockInput])
    x4 = Conv2D(filters=56, kernel_size=3, strides=1, padding='same', activation='relu')(x4)
    x4 = MG_Conv(x1, 32, (1, 1), activation = activation)
    x4 = MG_Conv(x1, 8, (3, 3), activation = activation)
    
    x5 = concatenate([blockInput, x1, x2, x3, x4])
    x5 = MG_Conv(x5, num_filters, (3, 3), activation = activation)
    
    x6 = Add()([blockInput, x5])
    
    return x6
    
    
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

"""
upscale_factor = 2
L = 2.5


net = MG_dense(inputs, num_filters = 64, l = 4)
#net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
#net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)

#net = MG_recur(net, num_filters = 16, batch_activate = False, recur_time = 3)
#net = MaxPooling2D(pool_size=2, padding='same')(net)
#net = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='relu')(net)
#net = Add()([inputs, net])
outputs = Activation('relu')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse',  metrics=['acc', 'mse'])

model.summary()
"""

upscale_factor = 2
L = 2.5

inputs = Input(shape=(40, 40, 3))

net1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
net2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net1)

densenet1 = MG_dense(net2, num_filters = 64, activation = False)
densenet2 = MG_dense(densenet1, num_filters = 64, activation = False)
densenet3 = MG_dense(densenet2, num_filters = 64, activation = False)

net3 = concatenate([densenet1, densenet2, densenet3])
net4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(net3)
net5 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(net4)
net6 = Add()([net1, net5])

#net7 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(net6)
net8 = Subpixel(filters=19, kernel_size=3, r=int(L*upscale_factor), padding='same')(net6)
#net = Conv2D(filters=6, kernel_size=3, strides=1, padding='same', activation='relu')(net8)
net = Conv2D(filters=3, kernel_size=3, strides=2, padding='same', activation='relu')(net8)
net = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='relu')(net)

outputs = Activation('relu')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse',  metrics=['acc', 'mse'])

model.summary()

from keras.utils import plot_model
plot_model(model, to_file = 'residual_dense_model_191026_nobatch.png')

#Train

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=64, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2, mode='auto', min_lr=1e-05)
])

model.save('D:\\microscopy\\training_DRN_191026_nobatch_sub7.h5')
print("Saved model to disk")

#Evaluation
model.save_weights('D:\\microscopy\\weights_training_DRN_191026_nobatch_sub_7.h5')


from keras_self_attention import SeqSelfAttention

model = load_model('C:\\Users\\ganze\\Desktop\\DRNlayer\\training_DRN_191026_nobatch_sub7.h5',custom_objects={'Subpixel': Subpixel})

model.load_weights('C:\\Users\\ganze\\Desktop\\DRNlayer\\weights_training_DRN_191026_nobatch_sub_19.h5')

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')

preds = model.predict(x_val)


from keras.preprocessing import image

img_path = 'C:\\Users\\ganze\\Desktop\\DRNlayer\\x_val\\x_val.png'
img = image.load_img(img_path, target_size=(40, 40))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# 모델이 훈련될 때 입력에 적용한 전처리 방식을 동일하게 사용합니다
img_tensor /= 255.

# 이미지 텐서의 크기는 (1, 150, 150, 3)입니다
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

from keras import models

# 상위 8개 층의 출력을 추출합니다:
layer_outputs = [layer.output for layer in model.layers][1:]
# 입력에 대해 8개 층의 출력을 반환하는 모델을 만듭니다:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)


plt.matshow(first_layer_activation[0, :, :, 19], cmap='viridis')
plt.show()

plt.matshow(first_layer_activation[0, :, :, 15], cmap='viridis')
plt.show()


# 층의 이름을 그래프 제목으로 사용합니다
layer_names = []
for layer in model.layers[1:]:
    layer_names.append(layer.name)

images_per_row = 16

# 특성 맵을 그립니다
for layer_name, layer_activation in zip(layer_names, activations):
    # 특성 맵에 있는 특성의 수
    n_features = layer_activation.shape[-1]

    # 특성 맵의 크기는 (1, size, size, n_features)입니다
    size = layer_activation.shape[1]

    # 활성화 채널을 위한 그리드 크기를 구합니다
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 각 활성화를 하나의 큰 그리드에 채웁니다
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # 그래프로 나타내기 좋게 특성을 처리합니다
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # 그리드를 출력합니다
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

#######################################시각화

###############필터 시각화


from keras import backend as K

model.load_weights('C:\\Users\\ganze\\Desktop\\DRNlayer\\weights_training_DRN_191026_nobatch_sub_19.h5')
layer_name = 'subpixel_1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# gradients 함수가 반환하는 텐서 리스트(여기에서는 크기가 1인 리스트)에서 첫 번째 텐서를 추출합니다
grads = K.gradients(loss, model.input)[0]

# 0 나눗셈을 방지하기 위해 1e–5을 더합니다
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

# 테스트:
loss_value, grads_value = iterate([np.zeros((1, 40, 40, 3))])


# 잡음이 섞인 회색 이미지로 시작합니다
input_img_data = np.random.random((1, 40, 40, 3)) * 20 + 128.

# 업데이트할 그래디언트의 크기
step = 1.
for i in range(40):   # 경사 상승법을 40회 실행합니다
    # 손실과 그래디언트를 계산합니다
    loss_value, grads_value = iterate([input_img_data])
    # 손실을 최대화하는 방향으로 입력 이미지를 수정합니다
    input_img_data += grads_value * step


def deprocess_image(x):
    # 텐서의 평균이 0, 표준 편차가 0.1이 되도록 정규화합니다
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1]로 클리핑합니다
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB 배열로 변환합니다
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=40):
    # 주어진 층과 필터의 활성화를 최대화하기 위한 손실 함수를 정의합니다
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 손실에 대한 입력 이미지의 그래디언트를 계산합니다
    grads = K.gradients(loss, model.input)[0]

    # 그래디언트 정규화
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 입력 이미지에 대한 손실과 그래디언트를 반환합니다
    iterate = K.function([model.input], [loss, grads])
    
    # 잡음이 섞인 회색 이미지로 시작합니다
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 경사 상승법을 40 단계 실행합니다
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


plt.imshow(generate_pattern('subpixel_1', 0))
plt.show()


for layer_name in ['conv2d_1', 'concatenate_4', 'add_1', 'concatenate_8', 'add_2', 'concatenate_12', 'concatenate_13', 'add_4', 'subpixel_1' ]:
    size = 64
    margin = 5

    # 결과를 담을 빈 (검은) 이미지
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), dtype='uint8')

    for i in range(8):  # results 그리드의 행을 반복합니다
        for j in range(8):  # results 그리드의 열을 반복합니다
            # layer_name에 있는 i + (j * 8)번째 필터에 대한 패턴 생성합니다
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # results 그리드의 (i, j) 번째 위치에 저장합니다
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # results 그리드를 그립니다
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()






"""
indx = 23

x_val_re = cv.normalize(x_val[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
y_val_re = cv.normalize(y_val[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
preds_re = cv.normalize(preds[indx].astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
preds_post = cv.normalize(preds[indx].astype(np.float64), None, 0, 330, cv.NORM_MINMAX)


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

cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\x_val\\x_val.png', img_x)

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
cv.imwrite('C:\\Users\\MIT-DGMIF\\Desktop\\denseunet\\training_result_image\\y_val\\y_val.png', img_y)

#----------------------------------------------------#


cv.imwrite('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds_nobatch_sub19\\predsnon.png', preds_post)
cv.imwrite('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds_nobatch_sub19\\predsnon_ori.png', preds_re)

preds_R = cv.imwrite('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\preds_R.png', preds_re[:,:,0])
preds_G = cv.imwrite('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\preds_G.png', preds_re[:,:,1])
preds_B = cv.imwrite('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\preds_B.png', preds_re[:,:,2])

preds_r = cv.imread('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\preds_R.png', cv.IMREAD_GRAYSCALE)
preds_g = cv.imread('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\preds_G.png', cv.IMREAD_GRAYSCALE)
preds_b = cv.imread('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\preds_B.png', cv.IMREAD_GRAYSCALE)

img_preds = cv.merge([preds_r, preds_g, preds_b])
plt.imshow(img_preds)
plt.show()
cv.imwrite('C:\\Users\\ganze\\Desktop\\DRNlayer\\preds\\prds.png', img_preds)


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

"""