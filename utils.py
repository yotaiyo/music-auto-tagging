import numpy as np
import pickle

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

def calcurate_num_samples(df, split):
    num = 0
    for index in df.index:
        df_split = df.split[index]
        if df_split == split:
            num += 1
    num *= 10
    return num