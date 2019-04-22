import sys
import os
import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

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

def calcurate_num_test(df):
    num_test = 0
    for index in df.index:
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == 'test':
            num_test += 1
    num_test *= 10
    return num_test

def calcurate_roc_auc(df, num_test, num_segment, feature_name_list, feature_size_dict, model):
    y_pred = np.zeros((num_test,50))
    y_true = np.zeros((num_test,50))
    count = 0
    for index in range(1,1000):
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == 'test':
            print('index',index)
            predict_label_sum = np.zeros(50)
            for j in range(1,11):
                feature_dir = 'features_norm/feature_%s_%d_%d.pickle' %(split,clip_id,j)
                features = load_feature(feature_dir)

                feature_list = []
                for feature_name in feature_name_list:
                    feature_size = feature_size_dict[feature_name]
                    feature = features[feature_name]
                    if len(feature_size) == 1:
                        feature = feature.reshape(1, feature_size[0])
                    if len(feature_size) == 2:
                        feature = feature.reshape(1, feature_size[0], feature_size[1])
                    feature_list.append(feature)
                
                predict_label = model.predict(feature_list).reshape(50)

                predict_label_sum += predict_label

            predict_label_song = predict_label_sum/num_segment
            true_label_song = df.iloc[:,1:51][index:index+1].values.reshape(50)
            
            y_pred[count] = predict_label_song
            y_true[count] = true_label_song
            
            count += 1

    roc_auc = roc_auc_score(y_true, y_pred,average='macro')
    return roc_auc

if __name__=='__main__':
    if len(sys.argv) == 4:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)

        num_test = calcurate_num_test(df)

        feature_name_list = ['zerocross','rms','pulseclarity','beatspectrum','flux','centroid','rolloff','flatness','entropy','skewness','kurtosis','chromagram','mfcc']

        feature_size_dict = load_feature_size(df, feature_name_list)

        num_segment = int(sys.argv[2])

        model_path = sys.argv[3]

        model = load_model(model_path)

        roc_auc = calcurate_roc_auc(df, num_test, num_segment, feature_name_list, feature_size_dict, model)
        print('your model roc_auc is %d' %roc_auc)

    else:
        print('usage: python eval.py input_path:annotations_final_top_50_tag.csv num_segment:10 model_path:model/my_model_epoch-val_loss.hdf5')



