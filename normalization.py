import sys
import pandas as pd
import numpy as np
import pickle

def normalization(df, feature_name_list, mean_dict, std_dict):
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
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)
        
        feature_name_list = ['zerocross','rms','pulseclarity','beatspectrum','flux','centroid','rolloff','flatness','entropy','skewness','kurtosis','chromagram','mfcc']

        with open('feature_mean_std/mean.pickle',mode='rb') as f:
            mean_dict = pickle.load(f)
        with open('feature_mean_std/std.pickle',mode='rb') as f:
            std_dict = pickle.load(f)

        normalization(df, feature_name_list, mean_dict, std_dict)
    else:
        print('usage: python normalization.py input_path:annotations_final_top_50_tag.csv')
