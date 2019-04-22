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

print('keras_version', keras.__version__)
print('tf.__version__', tf.__version__)

def load_feature_size(df, feature_name_list):
    index = 0
    split = df.split[index]
    clip_id = df.clip_id[index]
    feature_size_dict = {}
    for feature_name in feature_name_list:
        feature_dir = 'features/%s_%s_%d_1.txt' %(feature_name,split,clip_id)
        feature = np.loadtxt(feature_dir,skiprows=1)
        feature_size_dict[feature_name] = feature.shape
    return feature_size_dict

def load_feature(feature_dir):
    with open('%s' %feature_dir, 'rb') as f:
        feature = pickle.load(f)
    return feature

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

def create_input_and_output_list_of_all_features(feature_size_dict, filters, num_blocks, weight_decay):
    zerocross_input = Input(shape=(feature_size_dict['zerocross'][0],),name='zerocross')
    zerocross_output = create_feature_output_1D(zerocross_input, feature_size_dict['zerocross'], filters, num_blocks, weight_decay)

    rms_input = Input(shape=(feature_size_dict['rms'][0],),name='rms')
    rms_output = create_feature_output_1D(rms_input, feature_size_dict['rms'], filters, num_blocks, weight_decay)

    pulseclarity_input = Input(shape=(feature_size_dict['pulseclarity'][0],),name='pulseclarity')
    pulseclarity_output = create_feature_output_1D(pulseclarity_input, feature_size_dict['pulseclarity'], filters, num_blocks, weight_decay)

    beatspectrum_input = Input(shape=(feature_size_dict['beatspectrum'][0],),name='beatspectrum')
    beatspectrum_output = create_feature_output_1D(beatspectrum_input, feature_size_dict['beatspectrum'], filters, num_blocks, weight_decay)

    flux_input = Input(shape=(feature_size_dict['flux'][0],),name='flux')
    flux_output = create_feature_output_1D(flux_input, feature_size_dict['flux'], filters, num_blocks, weight_decay)

    centroid_input = Input(shape=(feature_size_dict['centroid'][0],),name='centroid')
    centroid_output = create_feature_output_1D(centroid_input, feature_size_dict['centroid'], filters, num_blocks, weight_decay)

    rolloff_input = Input(shape=(feature_size_dict['rolloff'][0],),name='rolloff')
    rolloff_output = create_feature_output_1D(rolloff_input, feature_size_dict['rolloff'], filters, num_blocks, weight_decay)

    flatness_input = Input(shape=(feature_size_dict['flatness'][0],),name='flatness')
    flatness_output = create_feature_output_1D(flatness_input, feature_size_dict['flatness'], filters, num_blocks, weight_decay)

    entropy_input = Input(shape=(feature_size_dict['entropy'][0],),name='entropy')
    entropy_output = create_feature_output_1D(entropy_input, feature_size_dict['entropy'], filters, num_blocks, weight_decay)

    skewness_input = Input(shape=(feature_size_dict['skewness'][0],),name='skewness')
    skewness_output = create_feature_output_1D(skewness_input, feature_size_dict['skewness'], filters, num_blocks, weight_decay)

    kurtosis_input = Input(shape=(feature_size_dict['kurtosis'][0],),name='kurtosis')
    kurtosis_output = create_feature_output_1D(kurtosis_input, feature_size_dict['kurtosis'], filters, num_blocks, weight_decay)

    chromagram_input = Input(shape=(feature_size_dict['chromagram'][0],feature_size_dict['chromagram'][1], ),name='chromagram')
    chromagram_output = create_feature_output_2D(chromagram_input, feature_size_dict['chromagram'], filters, num_blocks, weight_decay)

    mfcc_input = Input(shape=(feature_size_dict['mfcc'][0],feature_size_dict['mfcc'][1], ),name='mfcc')
    mfcc_output = create_feature_output_2D(mfcc_input, feature_size_dict['mfcc'], filters, num_blocks, weight_decay)

    feature_input_list = [zerocross_input,rms_input,pulseclarity_input,beatspectrum_input,flux_input,
                           centroid_input,rolloff_input,flatness_input,entropy_input,skewness_input,
                           kurtosis_input,chromagram_input,mfcc_input]

    feature_output_list = [zerocross_output,rms_output,pulseclarity_output,beatspectrum_output,flux_output,
                            centroid_output,rolloff_output,flatness_output,entropy_output,skewness_output,
                            kurtosis_output,chromagram_output,mfcc_output]

    return feature_input_list, feature_output_list

def create_predicted_value_by_fc_layer(feature_output_list, feature_size_dict, feature_name_list, filters, num_blocks, weight_decay, dropout_rate):
    feature_output_concatenated = concatenate(feature_output_list)
    x = Dense(feature_output_concatenated.shape[-1].value)(feature_output_concatenated)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    predicted_value = Dense(50, activation='sigmoid')(x)
    return predicted_value

def calcurate_num_train_and_val_sample(df):
    count_train = 0
    for index in df.index:
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == 'train':
            count_train += 1
    count_train *= 10

    count_val = 0
    for index in df.index:
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == 'val':
            count_val += 1
    count_val *= 10
    return count_train, count_val

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

    count_train, count_val = calcurate_num_train_and_val_sample(df)

    train_gen = generator(batch_size,df_train,labels_train,index_max_train,index_list_train, feature_name_list, feature_size_dict)

    val_gen = generator(batch_size,df_val,labels_val,index_max_val,index_list_val, feature_name_list, feature_size_dict)

    steps_per_epoch = math.ceil(count_train/batch_size)

    validation_steps = math.ceil(count_val/batch_size)

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
        print('usage: python train.py input_path:annotations_final_top_50_tag.csv, filters:32, num_blocks:3, weight_decay:0.001, dropout_rate:0.5, learning_rate:0.01, batch_size:23, epochs:100')
