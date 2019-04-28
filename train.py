import sys
import os
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import keras
import random
import math
import os
from keras.layers import (Conv1D,MaxPool1D,Conv2D,MaxPool2D,BatchNormalization,GlobalMaxPool1D,GlobalMaxPool2D,
                          Dense,Dropout,Activation,Reshape,Input,concatenate,
                          Multiply,GlobalAvgPool1D,GlobalAvgPool2D,Add)
from keras import Input 
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers
import time
from utils import load_feature_size
from utils import load_feature
from utils import calcurate_num_samples 

print('keras_version', keras.__version__)
print('tf.__version__', tf.__version__)

def load_feature_dir(split_name):
    feature_dir_dir = 'features_dir/feature_%s_dir.csv' %(split_name)
    df_feature = pd.read_csv(feature_dir_dir)
    return df_feature

def basic_block_conv1D(x, filters, kernel_size, weight_decay, pool_size):
    x = Conv1D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size)(x)
    return x

def basic_block_conv2D(x, filters, kernel_size, wight_decay, pool_size):
    x = Conv2D(filters, kernel_size, strides=(1,1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size)(x)
    return x

def calcurate_kernel_size_and_pool_size(feature_size, num_blocks):
    if num_blocks <= 1:
        print('danger! num_blocksは1より大きくなければならない')
    flag = True
    count = 1
    while flag:
        if count ** num_blocks < feature_size:
            count += 1
        else:
            count -= 1 
            flag = False
    return count

def create_feature_output_1D(feature_input, feature_size, filters, num_blocks, weight_decay):
    x = Reshape([-1,1])(feature_input)
    kernel_size = calcurate_kernel_size_and_pool_size(feature_size[0], num_blocks)
    pool_size = calcurate_kernel_size_and_pool_size(feature_size[0], num_blocks)

    for i in range(num_blocks):
        if i % (num_blocks // 3) == 0:
            filters *= 2
        x = basic_block_conv1D(x, filters, kernel_size, weight_decay, pool_size)

    feature_output = GlobalMaxPool1D()(x)
    return feature_output

def create_feature_output_2D(feature_input, feature_size, filters, num_blocks, weight_decay):
    x = Reshape([feature_size[0] ,feature_size[1], 1])(feature_input)

    size_0 = calcurate_kernel_size_and_pool_size(feature_size[0], num_blocks)
    size_1 = calcurate_kernel_size_and_pool_size(feature_size[1], num_blocks)

    kernel_size = (size_0, size_1)
    pool_size = (size_0, size_1)

    for i in range(num_blocks):
        if i % (num_blocks // 3) == 0:
            filters *= 2
        x = basic_block_conv2D(x, filters, kernel_size, weight_decay, pool_size)

    feature_output = GlobalMaxPool2D()(x)
    return feature_output

def create_input_and_output_list_of_all_features(feature_name_list, feature_size_dict, filters, num_blocks, weight_decay):
    feature_input_list = []
    feature_output_list = []
    for feature_name in feature_name_list:
        feature_size = feature_size_dict[feature_name]
        if len(feature_size) == 1:
            feature_input = Input(shape=(feature_size_dict[feature_name][0],),name=feature_name)
            feature_output = create_feature_output_1D(feature_input, feature_size_dict[feature_name], filters, num_blocks, weight_decay)
        elif len(feature_size) == 2:
            feature_input = Input(shape=(feature_size_dict[feature_name][0],feature_size_dict[feature_name][1], ),name=feature_name)
            feature_output = create_feature_output_2D(feature_input, feature_size_dict[feature_name], filters, num_blocks, weight_decay)

        feature_input_list.append(feature_input)
        feature_output_list.append(feature_output)

    return feature_input_list, feature_output_list

def create_predicted_value_by_fc_layer(feature_output_list, feature_size_dict, feature_name_list, filters, num_blocks, weight_decay, dropout_rate):
    feature_output_concatenated = concatenate(feature_output_list)
    x = Dense(feature_output_concatenated.shape[-1].value)(feature_output_concatenated)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    predicted_value = Dense(50, activation='sigmoid')(x)
    return predicted_value

def generator(batch_size, df, labels, index_max, index_list, feature_name_list, feature_size_dict):
    while 1:
        indices = random.sample(index_list,batch_size)

        feature_batch_list = []
        for feature_name in feature_name_list:

            feature_size = feature_size_dict[feature_name] 

            if len(feature_size) == 1:
                feature_batch = np.zeros((batch_size,feature_size[0]))
            elif len(feature_size) == 2:
                feature_batch = np.zeros((batch_size,feature_size[0],feature_size[1]))

            labels_batch = np.zeros((batch_size,50))
            count = 0

            for batch_index in indices:
                
                df_dir = df.features_dir[batch_index]
                feature = load_feature(df_dir)

                feature_batch[count] = feature[feature_name]

                label = labels[batch_index]
                labels_batch[count] = label

                count += 1
            
            feature_batch_list.append(feature_batch)
        
        yield feature_batch_list, labels_batch

def step_decay(epoch):
    x = 0.01
    if epoch >= 20: 
        x = 0.002
    if epoch >= 40: 
        x = 0.0004
    if epoch >= 60: 
        x = 0.00008
    if epoch >= 80: 
        x = 0.000016
    if epoch >= 100: 
        x = 0.0000032
    if epoch >= 120: 
        x = 0.00000064
    if epoch >= 140: 
        x = 0.000000128    
    return x

def train_model(df, feature_name_list, feature_size_dict, filters, num_blocks, weight_decay, dropout_rate, learning_rate, batch_size, epochs):
    df_train = load_feature_dir('train')
    labels_train = df_train.iloc[:,1:51].values
    index_max_train = df_train.index.max()
    index_list_train = list(range(index_max_train))

    df_val = load_feature_dir('val')
    labels_val = df_val.iloc[:,1:51].values
    index_max_val = df_val.index.max()
    index_list_val = list(range(index_max_val))

    num_train = calcurate_num_samples(df, 'train')

    num_val = calcurate_num_samples(df, 'val')

    train_gen = generator(batch_size,df_train,labels_train,index_max_train,index_list_train, feature_name_list, feature_size_dict)

    val_gen = generator(batch_size,df_val,labels_val,index_max_val,index_list_val, feature_name_list, feature_size_dict)

    steps_per_epoch = math.ceil(num_train/batch_size)

    validation_steps = math.ceil(num_val/batch_size)

    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('train_log'):
        os.mkdir('train_log')
    if not os.path.exists('my_log_dir'):
        os.mkdir('my_log_dir')

    feature_input_list, feature_output_list = create_input_and_output_list_of_all_features(feature_size_dict, filters, num_blocks, weight_decay)

    predicted_value = create_predicted_value_by_fc_layer(feature_output_list, feature_size_dict, feature_name_list, filters, num_blocks, weight_decay, dropout_rate)

    model = Model(feature_input_list, predicted_value)
    sgd = optimizers.SGD(lr=learning_rate,momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['mae'])
    print('model is created!')
    model.summary()

    callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='model/my_model_{epoch:03d}-{val_loss:.4f}.hdf5',
                                                      monitor='val_loss',
                                                      save_best_only=True),
                  keras.callbacks.CSVLogger('train_log/training.csv', append=True),
                  keras.callbacks.TensorBoard(log_dir='my_log_dir'),
                  keras.callbacks.LearningRateScheduler(step_decay)]

    print('start train your model')
    start = time.time()
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks_list)
    processtime = time.time() - start
    print('finish train your model')
    print('processtime is %s' %processtime)


if __name__=='__main__':
    if len(sys.argv) == 9:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)

        feature_name_list = ['zerocross','rms','pulseclarity','beatspectrum','flux','centroid','rolloff','flatness','entropy','skewness','kurtosis','chromagram','mfcc']

        feature_size_dict = load_feature_size(df, feature_name_list)

        filters = int(sys.argv[2])
        num_blocks = int(sys.argv[3])
        weight_decay = float(sys.argv[4])
        dropout_rate = float(sys.argv[5])
        learning_rate = float(sys.argv[6])
        batch_size = int(sys.argv[7])
        epochs = int(sys.argv[8])

        train_model(df, feature_name_list, feature_size_dict, filters, num_blocks, weight_decay, dropout_rate, learning_rate, batch_size, epochs)
    
    else:
        print('usage: python train.py input_path:annotations_final_top_50_tag.csv filters:32 num_blocks:3 weight_decay:0.001 dropout_rate:0.5 learning_rate:0.01 batch_size:23 epochs:100')
