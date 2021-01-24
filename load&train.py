# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:53:11 2019

@author: admin
"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import hsi_net
import spectral
import scipy.io as sio
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn import metrics, preprocessing
from Utils import zeroPadding
import averageAccuracy
from keras.models import load_model
import Kappa
import collections
import time

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)

    for i in range(m):
#        print(i)
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
#        print(indices)

        labels_loc[i] = indices
        nb_val = int((1-proptionVal) * len(indices))
        print('class :',i,'num :',nb_val,'len :',int(len(indices)))
        #print(nb_val)
        train[i] = indices[:nb_val+2]
        test[i] = indices[nb_val+2:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
    print(len(test_indices))



mat_data = sio.loadmat('dataset/IN/Indian_pines_corrected.mat')
print(mat_data.keys())
data_IN = mat_data['indian_pines_corrected']
print(data_IN.shape)

mat_data1 = sio.loadmat('dataset/IN/Indian_pines_10.mat')
print(mat_data1.keys())
data_IN1=mat_data1['data']
print(data_IN1.shape)

mat_gt = sio.loadmat('dataset/IN/Indian_pines_gt.mat')
print(mat_gt.keys())
gt_IN = mat_gt['indian_pines_gt']
print(gt_IN.shape)



new_gt_IN = gt_IN


nb_classes = 16

##############################################################
# ####### here change your patch size
img_rows, img_cols = 9,9
img_rows1,img_cols1=27,27

####here change  dimension

##########################################

# 3%:5%:92% data for training, validation and testing
TOTAL_SIZE = 10249
VAL_SIZE =520
TRAIN_SIZE = 330
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
ALL_SIZE = data_IN.shape[0] * data_IN.shape[1]
##############################################################
VALIDATION_SPLIT = 0.97
##############################################################

PATCH_LENGTH = 4
PATCH_LENGTH1 = 13

INPUT_DIMENSION_CONV = 200
INPUT_DIMENSION_CONV1 = 10

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
data1 = data_IN1.reshape(np.prod(data_IN1.shape[:2]), np.prod(data_IN1.shape[2:]))

gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
print(gt.shape)

data = preprocessing.scale(data)
data1 = preprocessing.scale(data1)


# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
data_1 = data1.reshape(data_IN1.shape[0], data_IN1.shape[1],data_IN1.shape[2])

whole_data = data_
whole_data1=data_1

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
padded_data1 = zeroPadding.zeroPadding_3D(whole_data1, PATCH_LENGTH1)


ITER = 1
###############################################
CATEGORY = 16

train_data = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
train_data1 = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH1 + 1, 2*PATCH_LENGTH1 + 1, INPUT_DIMENSION_CONV1))


test_data = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data1 = np.zeros((TEST_SIZE, 2*PATCH_LENGTH1 + 1, 2*PATCH_LENGTH1 + 1, INPUT_DIMENSION_CONV1))


#seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

#seeds = [1334]

seeds=[1220]

NUM=1
for num in range(NUM):
    for index_iter in range(ITER):
        print("# %d Iteration" % (index_iter + 1))

    
        np.random.seed(seeds[index_iter])

        train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
        m=len(test_indices)
        n=len(train_indices)
        print('test',m)
        print('train',n)
       

        y_train = gt[train_indices] - 1
        y_train = to_categorical(np.asarray(y_train))

        y_test = gt[test_indices] - 1
        gt_test=y_test[:-VAL_SIZE]
        y_test = to_categorical(np.asarray(y_test))

  
        train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        train_assign1 = indexToAssignment(train_indices, whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)
        
        #np.random.seed(None)
        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)
          
        for i in range(len(train_assign1)):
            train_data1[i] = selectNeighboringPatch(padded_data1, train_assign1[i][0], train_assign1[i][1], PATCH_LENGTH1)
            
        test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        test_assign1 = indexToAssignment(test_indices, whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)
        # sess2=tf.Session()
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)
        
        for i in range(len(test_assign)):
            test_data1[i] = selectNeighboringPatch(padded_data1, test_assign1[i][0], test_assign1[i][1], PATCH_LENGTH1)
            
        all_assign = indexToAssignment(range(ALL_SIZE), whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        all_assign1 = indexToAssignment(range(ALL_SIZE), whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)
        
            
        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
        x_train1 = train_data1.reshape(train_data1.shape[0], train_data1.shape[1], train_data1.shape[2], INPUT_DIMENSION_CONV1)

        x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)
        x_test_all1 = test_data1.reshape(test_data1.shape[0], test_data1.shape[1], test_data1.shape[2], INPUT_DIMENSION_CONV1)
        

        x_val = x_test_all[-VAL_SIZE:]
        x_val1 = x_test_all1[-VAL_SIZE:]
        y_val = y_test[-VAL_SIZE:]

        x_test = x_test_all[:-VAL_SIZE]
        x_test1 = x_test_all1[:-VAL_SIZE]
        y_test = y_test[:-VAL_SIZE]
        
        
      
        print("x_train shape :",x_train.shape,x_train1.shape)  # (228, 9, 9, 200) (228, 27, 27, 10)
        print("y_train shape :",y_train.shape)
        print('x_val shape :',x_val.shape,x_val1.shape)  # (520, 9, 9, 200) (520, 27, 27, 10)
        print('y_val shape :',y_val.shape)
        print("x_test shape :",x_test.shape,x_test1.shape)
        print("y_test shape :",y_test.shape)

train_xh27=x_train1.reshape(-1,27,27,10)
train_xhv27=x_val1.reshape(-1,27,27,10)
train_xh=x_train.reshape(-1,9,9,200)
train_xhv=x_val.reshape(-1,9,9,200)
test_hsi27_x=x_test_all1.reshape(-1,27,27,10)
test_hsi9_x=x_test_all.reshape(-1,9,9,200)

########################MFDN##########################
model=hsi_net.Network()
print('training+++++++++++++++++++++++++')
model_ckt = ModelCheckpoint(filepath='./ckpt/best.h5', verbose=1, save_best_only=True)
train_begin=time.clock()
model.fit([train_xh27, train_xh], y_train, epochs=200, batch_size=25,
          callbacks=[model_ckt], validation_data=([train_xhv27,train_xhv], y_val))
train_end=time.clock()
scores = model.evaluate([train_xhv27,train_xhv],y_val,batch_size=25)
print('Test score:', scores[0])
print('Test accuracy:', scores[1])
# model.save_weights('model/italy_60epoch-5%.h5')

model.load_weights('./ckpt/best.h5')
test_begin=time.clock()
pred=model.predict([test_hsi27_x,test_hsi9_x]).argmax(axis=1)
test_end=time.clock()

collections.Counter(pred)
gt_test = gt[test_indices] - 1
gt_train = gt[train_indices] - 1

overaccy=metrics.accuracy_score(pred,gt_test)
confusion_matrix_mss=metrics.confusion_matrix(gt_test,pred)
print(confusion_matrix_mss)
average_acc=averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_mss)
kappa_value=Kappa.kappa(confusion_matrix_mss)
print('\nOA:',overaccy)
print('\nAA:',average_acc)
print('\nKappa:',kappa_value)

print('train time :',train_end-train_begin)#time
print('test time :',test_end-test_begin)
collections.Counter(pred)
gt_test = gt[test_indices] - 1
gt_train = gt[train_indices] - 1
#pred_test_conv1=model.predict([x_pic1,x_pic]).argmax(axis=1)
#print(pred_test_conv1.shape)

####################draw pic###################
new_show = np.zeros((gt_IN.shape[0], gt_IN.shape[1]))
hsi_pic=np.zeros((gt_IN.shape[0],gt_IN.shape[1],3))
print(pred.shape[0])
for k in range(pred.shape[0]):
    n = test_indices[k]
    i = int(n / new_show.shape[1])
    j = n - i * new_show.shape[1]
    new_show[i][j] = pred[k] + 1

for k in range(gt_train.shape[0]):
    n = train_indices[k]
    i = int(n / new_show.shape[1])
    j = n - i * new_show.shape[1]
    new_show[i][j] = gt_train[k] + 1
    
    
print(new_show.shape[0],print(new_show.shape[1]))
for i in range(new_show.shape[0]):
    for j in range(new_show.shape[1]):
        if new_show[i][j]==0:
            hsi_pic[i,j,:]=[0, 0, 0]
        if new_show[i][j]==1:
            hsi_pic[i,j,:]=[0, 0, 1]
        if new_show[i][j]==2:
            hsi_pic[i,j,:]=[0, 1, 0]
        if new_show[i][j]==3:
            hsi_pic[i,j,:]=[0, 1, 1]
        if new_show[i][j]==4:
            hsi_pic[i,j,:]=[1, 0, 0]
        if new_show[i][j]==5:
            hsi_pic[i,j,:]=[1, 0, 1]
        if new_show[i][j]==6:
            hsi_pic[i,j,:]=[1, 1, 0]
        if new_show[i][j]==7:
            hsi_pic[i,j,:]=[0.5, 0.5, 1]
        if new_show[i][j]==8:
            hsi_pic[i,j,:]=[0.65, 0.35, 1]
        if new_show[i][j]==9:
            hsi_pic[i,j,:]=[0.75, 0.5, 0.75]
        if new_show[i][j]==10:
            hsi_pic[i,j,:]=[0.75, 1, 0.5]
        if new_show[i][j]==11:
            hsi_pic[i,j,:]=[0.5, 1, 0.65]
        if new_show[i][j]==12:
            hsi_pic[i,j,:]=[0.65, 0.65, 0]
        if new_show[i][j]==13:
            hsi_pic[i,j,:]=[0.75, 1, 0.65]
        if new_show[i][j]==14:
            hsi_pic[i,j,:]=[0, 0, 0.5]
        if new_show[i][j]==15:
            hsi_pic[i,j,:]=[0, 1, 0.75]
        if new_show[i][j]==16:
            hsi_pic[i,j,:]=[0.5, 0.75, 1]
classification_map(hsi_pic, gt_IN, 24, "RES4_SS_IN.png")    
color = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 1],
                          [0.65, 0.35, 1], [0.75, 0.5, 0.75], [0.75, 1, 0.5], [0.5, 1, 0.65], [0.65, 0.65, 0],
                          [0.75, 1, 0.65], [0, 0, 0.5], [0, 1, 0.75], [0.5, 0.75, 1]])
color = color * 255

pre = spectral.imshow(classes=new_show.astype(int), figsize=(9, 9), colors=color)