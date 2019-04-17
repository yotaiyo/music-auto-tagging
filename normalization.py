import sys
import numpy as np
import pandas as pd
import pickle
import os

def calcurate_mean(feature_name, df):
    if feature_name == 'pulseclarity':
        feature_sum = np.zeros(27)
    elif feature_name == 'beatspectrum':
        feature_sum = np.zeros(243)
    elif feature_name == 'flux':
        feature_sum = np.zeros(243)
    elif feature_name == 'chromagram':
        feature_sum = np.zeros((243,12))
    elif feature_name == 'mfcc':
        feature_sum = np.zeros((243,23))
    else:
        feature_sum = np.zeros(243)
    count = 0
    for index in df.index:
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

def calcurate_std(feature_name,feature_mean, df):
    if feature_name == 'pulseclarity':
        feature_squared_diff_sum = np.zeros(27)
    elif feature_name == 'beatspectrum':
        feature_squared_diff_sum = np.zeros(243)
    elif feature_name == 'flux':
        feature_squared_diff_sum = np.zeros(243)
    elif feature_name == 'chromagram':
        feature_squared_diff_sum = np.zeros((243,12))
    elif feature_name == 'mfcc':
        feature_squared_diff_sum = np.zeros((243,23))
    else:
        feature_squared_diff_sum = np.zeros(243)
    count = 0
    for index in df.index:
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
    
def calcurate_mean_and_std_of_all_features(df):
    zerocross_mean = calcurate_mean('zerocross', df)
    zerocross_std = calcurate_std('zerocross',zerocross_mean, df)

    rms_mean = calcurate_mean('rms', df)
    rms_std = calcurate_std('rms',rms_mean, df)

    pulseclarity_mean = calcurate_mean('pulseclarity', df)
    pulseclarity_std = calcurate_std('pulseclarity',pulseclarity_mean, df)

    beatspectrum_mean = calcurate_mean('beatspectrum', df)
    beatspectrum_std = calcurate_std('beatspectrum',beatspectrum_mean, df)

    flux_mean = calcurate_mean('flux', df)
    flux_std = calcurate_std('flux',flux_mean, df)

    centroid_mean = calcurate_mean('centroid', df)
    centroid_std = calcurate_std('centroid',centroid_mean, df)

    rolloff_mean = calcurate_mean('rolloff', df)
    rolloff_std = calcurate_std('rolloff',rolloff_mean, df)

    flatness_mean = calcurate_mean('flatness', df)
    flatness_std = calcurate_std('flatness',flatness_mean, df)

    entropy_mean = calcurate_mean('entropy', df)
    entropy_std = calcurate_std('entropy',entropy_mean, df)

    skewness_mean = calcurate_mean('skewness', df)
    skewness_std = calcurate_std('skewness',skewness_mean, df)

    kurtosis_mean = calcurate_mean('kurtosis', df)
    kurtosis_std = calcurate_std('kurtosis',kurtosis_mean, df)

    chromagram_mean = calcurate_mean('chromagram', df)
    chromagram_std = calcurate_std('chromagram',chromagram_mean, df)

    mfcc_mean = calcurate_mean('mfcc', df)
    mfcc_std = calcurate_std('mfcc',mfcc_mean, df)

    mean_dict = {}
    std_dict = {}
    for feature_name in feature_name_list:
        if feature_name == 'zerocross':
            mean_dict[feature_name] = zerocross_mean
            std_dict[feature_name] = zerocross_std
        elif feature_name == 'rms':
            mean_dict[feature_name] = rms_mean
            std_dict[feature_name] = rms_std
        elif feature_name == 'pulseclarity':
            mean_dict[feature_name] = pulseclarity_mean
            std_dict[feature_name] = pulseclarity_std
        elif feature_name == 'beatspectrum':
            mean_dict[feature_name] = beatspectrum_mean
            std_dict[feature_name] = beatspectrum_std
        elif feature_name == 'flux':
            mean_dict[feature_name] = flux_mean
            std_dict[feature_name] = flux_std
        elif feature_name == 'centroid':
            mean_dict[feature_name] = centroid_mean
            std_dict[feature_name] = centroid_std
        elif feature_name == 'rolloff':
            mean_dict[feature_name] = rolloff_mean
            std_dict[feature_name] = rolloff_std
        elif feature_name == 'flatness':
            mean_dict[feature_name] = flatness_mean
            std_dict[feature_name] = flatness_std
        elif feature_name == 'entropy':
            mean_dict[feature_name] = entropy_mean
            std_dict[feature_name] = entropy_std
        elif feature_name == 'skewness':
            mean_dict[feature_name] = skewness_mean
            std_dict[feature_name] = skewness_std
        elif feature_name == 'kurtosis':
            mean_dict[feature_name] = kurtosis_mean
            std_dict[feature_name] = kurtosis_std
        elif feature_name == 'chromagram':
            mean_dict[feature_name] = chromagram_mean
            std_dict[feature_name] = chromagram_std
        elif feature_name == 'mfcc':
            mean_dict[feature_name] = mfcc_mean
            std_dict[feature_name] = mfcc_std

    return mean_dict, std_dict

def save_mean_and_std(mean_dict, std_dict, feature_name_list):
    if not os.path.exists('feature_mean_std'):
        os.mkdir('feature_mean_std')

    with open('feature_mean_std/mean.pickle', 'wb') as f:
        pickle.dump(mean_dict, f)
        print('save mean succeed!')
    with open('feature_mean_std/std.pickle', 'wb') as f:
        pickle.dump(std_dict, f)
        print('save std succeed!')

def normalization(mean_dict, std_dict, feature_name_list, df):
    for index in df.index:
        print('index',index)
        split = df.split[index]
        clip_id = df.clip_id[index]
        for j in range(1,11):
            feature_dict = {}
            for feature_name in feature_name_list:
                feature_mean = mean_dict[feature_name]
                feature_std = std_dict[feature_name] 
                feature_dir = 'features/%s_%s_%d_%d.txt' %(feature_name,split,clip_id,j)
                feature = np.loadtxt(feature_dir,skiprows=1)
                feature[np.isnan(feature)] = 0
                feature_norm = (feature - feature_mean) / feature_std
                feature_dict[feature_name] = feature_norm
            with open('features_norm/feature_%s_%d_%d.pickle' %(split,clip_id,j), 'wb') as f:
                pickle.dump(feature_dict, f)

if __name__=='__main__':
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)

        feature_name_list = ['zerocross','rms','pulseclarity','beatspectrum','flux','centroid','rolloff','flatness','entropy','skewness','kurtosis','chromagram','mfcc']

        mean_dict, std_dict = calcurate_mean_and_std_of_all_features(df)
        save_mean_and_std(mean_dict, std_dict, feature_name_list)

        normalization(mean_dict, std_dict, feature_name_list, df)
    else:
        print('usage: python normalization.py input_path:annotations_final_top_50_tag.csv')




