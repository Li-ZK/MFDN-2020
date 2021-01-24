# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:35:26 2019

@author: admin
"""

import numpy as np
import math
np.random.seed(1337)
from keras.layers import Conv3D,BatchNormalization,concatenate,Flatten,Dropout,Dense,AveragePooling3D
from keras.layers import Conv2D,AveragePooling2D,MaxPool2D,Add,Activation
from keras.layers.advanced_activations import PReLU,LeakyReLU,ReLU
from keras.layers import Input
from keras.layers.core import Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

chn=200
class_num=16
#def spatital(hsi_img):
#    hsi_conv=Conv2D(20,(4,4),padding='valid',strides=(1,1))(hsi_img)
#    hsi_act=ReLU()(hsi_conv)
#    hsi_pool=AveragePooling2D((2,2))(hsi_act)  
#    hsi_pool=Dense_block_2D(hsi_pool,(4,4),20,18) 
#    conv1=Conv2D(74,(3,3),padding='valid',strides=(1,1))(hsi_pool)
#    act1=ReLU()(conv1)
#    pool1=AveragePooling2D((2,2))(act1)
#    pool1=Dense_block_2D(pool1,(4,4),74,18)
#    conv2=Conv2D(128,(3,3),padding='valid',strides=(1,1))(pool1)
#    act2=ReLU()(conv2) 
#    return act2
#
###############空间通道#############
def spatital(hsi_img):
    # hsi_img=BatchNormalization()(hsi_img)
    hsi_img=PReLU()(hsi_img)
    conv1=Conv2D(32,(3,3),padding='same')(hsi_img)
    pool1=AveragePooling2D((3,3))(conv1)
    
    dense_layer=Dense_block_2D(pool1,(3,3),32,32)
    dense_layer=BatchNormalization()(pool1)
    dense_layer=PReLU()(dense_layer)
    conv2=Conv2D(128,(3,3),padding='same')(dense_layer)
    pool2=AveragePooling2D((3,3))(conv2)
    return pool2

def Dense_block_2D(input_tensor,kernel_size,kernel_num,add_kernel_num):   
    layer1_BN=BatchNormalization()(input_tensor)
    layer1_PRelu=PReLU()(layer1_BN)
    layer1_conv=Conv2D(kernel_num+add_kernel_num,kernel_size,strides=(1,1),padding='same')(layer1_PRelu)
    layer2_input=concatenate([input_tensor,layer1_conv])
    layer2_BN=BatchNormalization()(layer2_input)
    layer2_PRelu=PReLU()(layer2_BN)
    layer2_conv=Conv2D(kernel_num+add_kernel_num*2,kernel_size,strides=(1,1),padding='same')(layer2_PRelu)
    layer3_input=concatenate([layer2_input,layer2_conv])
    layer3_BN=BatchNormalization()(layer3_input)
    layer3_PRelu=PReLU()(layer3_BN)
    layer3_conv=Conv2D(kernel_num+add_kernel_num*3,kernel_size,strides=(1,1),padding='same')(layer3_PRelu)
    output=concatenate([layer3_input,layer3_conv])    
    return output

def Dense_block(input_tensor,kernel_size,kernel_num):
    layer1_BN=BatchNormalization()(input_tensor)
    layer1_PRelu=PReLU()(layer1_BN)
    layer1_conv=Conv3D(kernel_num,kernel_size,strides=(1,1,1),padding='same')(layer1_PRelu)
    layer2_input=concatenate([input_tensor,layer1_conv])
    layer2_BN=BatchNormalization()(layer2_input)
    layer2_PRelu=PReLU()(layer2_BN)
    layer2_conv=Conv3D(kernel_num,kernel_size,strides=(1,1,1),padding='same')(layer2_PRelu)
    layer3_input=concatenate([layer2_input,layer2_conv])
    layer3_BN=BatchNormalization()(layer3_input)
    layer3_PRelu=PReLU()(layer3_BN)
    layer3_conv=Conv3D(kernel_num,kernel_size,strides=(1,1,1),padding='same')(layer3_PRelu)
    output=concatenate([layer3_input,layer3_conv])
    return output

################光谱通道################
def spectral(hsi_img):
    # hsi_img=BatchNormalization()(hsi_img)
    hsi_img=PReLU()(hsi_img)
    conv1=Conv3D(chn,(1,1,chn),padding='valid',strides=(1,1,1))(hsi_img)
    print('conv1 shape',conv1.shape)
    conv1=Reshape((conv1._keras_shape[1],conv1._keras_shape[2],conv1._keras_shape[4],conv1._keras_shape[3]))(conv1)  # (7,7,200,1)
    print(conv1.shape)
#    act1=PReLU()(conv1)
    dense_layer=Dense_block(conv1,(1,1,7),12)
    print('dense_layer shape is ',dense_layer.shape)
    dense_layer=BatchNormalization()(conv1)
    dense_layer=PReLU()(dense_layer)
    conv2=Conv3D(256,(1,1,chn),padding='valid')(dense_layer)
#    conv2=Conv3D(64,(1,1,29),padding='valid')(dense_layer)
    print('after dense conv shape is',conv2.shape)
    pool2=AveragePooling3D((3,3,1))(conv2)
    return pool2

###############MFDN#############
def Network():
    hsi27_input=Input(shape=(27,27,10))
    hsi9_input=Input(shape=(9,9,chn))
    hsi9_input1=Reshape((hsi9_input._keras_shape[1],hsi9_input._keras_shape[2],hsi9_input._keras_shape[3],1))(hsi9_input)

    spatital_feature=spatital(hsi27_input)
    spectral_feature=spectral(hsi9_input1)

    spatital_feature=Reshape((spatital_feature._keras_shape[1],spatital_feature._keras_shape[2],1,spatital_feature._keras_shape[3]))(spatital_feature)
    feature_fusion1=concatenate([spectral_feature,spatital_feature])
    feature_fusion1=Reshape((feature_fusion1._keras_shape[1],feature_fusion1._keras_shape[2],feature_fusion1._keras_shape[4],feature_fusion1._keras_shape[3]))(feature_fusion1)
    feature_fusion1=Dense_block(feature_fusion1,(3,3,7),16)
    
    print(feature_fusion1.shape)
    feature_fusion1=BatchNormalization()(feature_fusion1)
    feature_fusion1=PReLU()(feature_fusion1)
    feature_fusion1=Conv3D(256,(1,1,384),padding='valid')(feature_fusion1)
    print(feature_fusion1.shape)
    feature_fusion1=AveragePooling3D((3,3,1))(feature_fusion1)
    feature_fusion1=Flatten()(feature_fusion1)
    dropout_layer=Dropout(0.8)(feature_fusion1)
    dropout_layer=BatchNormalization()(dropout_layer)
    dropout_layer=Dense(256)(dropout_layer)
    dropout_layer=PReLU()(dropout_layer)
    pred=Dense(class_num,activation='softmax')(dropout_layer)
    model=Model([hsi27_input,hsi9_input],pred)
    adam=Adam(lr=0.0001)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['acc'])
    model.summary()
    return model
