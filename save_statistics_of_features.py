import sys
import numpy as np
import pandas as pd
import pickle
import os

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

def calcurate_mean(df, feature_name, feature_size):
    print('start calcurate %s mean' %feature_name)
    if len(feature_size) == 1:
        feature_sum = np.zeros(feature_size[0])
    elif len(feature_size) == 2:
        feature_sum = np.zeros((feature_size[0], feature_size[1]))
    count = 0
    for index in df.index:
        print('index',index)
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == 'train':
            for j in range(1,11):
                feature_dir = 'features/%s_%s_%d_%d.txt' %(feature_name,split,clip_id,j)
                feature = np.loadtxt(feature_dir,skiprows=1)
                feature[np.isnan(feature)] = 0
                feature_sum += feature
                count += 1
    feature_mean = (feature_sum / count).mean()
    return feature_mean

def calcurate_std(df, feature_name,feature_size, feature_mean):
    print('start calcurate %s std' %feature_name)
    if len(feature_size) == 1:
        feature_squared_diff_sum = np.zeros(feature_size[0])
    elif len(feature_size) == 2:
        feature_squared_diff_sum = np.zeros((feature_size[0], feature_size[1]))
    count = 0
    for index in df.index:
        print('index',index)
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == 'train':
            for j in range(1,11):
                feature_dir = 'features/%s_%s_%d_%d.txt' %(feature_name,split,clip_id,j)
                feature = np.loadtxt(feature_dir,skiprows=1)
                feature[np.isnan(feature)] = 0
                feature_diff = feature - feature_mean
                feature_squared_diff = feature_diff**2
                feature_squared_diff_sum += feature_squared_diff
                count += 1
    feature_std = (np.sqrt(feature_squared_diff_sum / count)).mean()
    return feature_std
    
def calcurate_mean_and_std_of_all_features(df, feature_name_list, feature_size_dict):
    mean_dict = {}
    std_dict = {}
    for feature_name in feature_name_list:
        print('start calcurate statistics of %s' %feature_name)
        feature_size = feature_size_dict[feature_name] 
        mean_dict[feature_name] = calcurate_mean(df, feature_name, feature_size)
        std_dict[feature_name] = calcurate_std(df, feature_name, feature_size, mean_dict[feature_name])
    return mean_dict, std_dict

def save_mean_and_std(feature_name_list, mean_dict, std_dict):
    if not os.path.exists('feature_mean_std'):
        os.mkdir('feature_mean_std')

    with open('feature_mean_std/mean.pickle', 'wb') as f:
        pickle.dump(mean_dict, f)
        print('save mean succeed!')
    with open('feature_mean_std/std.pickle', 'wb') as f:
        pickle.dump(std_dict, f)
        print('save std succeed!')

if __name__=='__main__':
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)

        feature_name_list = ['zerocross','rms','pulseclarity','beatspectrum','flux','centroid','rolloff','flatness','entropy','skewness','kurtosis','chromagram','mfcc']

        feature_size_dict = load_feature_size(df, feature_name_list)

        mean_dict, std_dict = calcurate_mean_and_std_of_all_features(df, feature_name_list, feature_size_dict)
        save_mean_and_std(feature_name_list, mean_dict, std_dict)
    else:
        print('usage: python save_statistics_of_features.py input_path:annotations_final_top_50_tag.csv')




