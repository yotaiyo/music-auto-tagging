import sys
import pandas as pd
import random
import os
import pickle

def create_features_norm_dir(df, split_name):
    print('start create %s features_norm_dir' %split_name)
    features_dir = []
    labels = []
    labels_column = df.iloc[:,1:51].columns.values.tolist()
    for index in df.index:
        print('index', index)
        split = df.split[index]
        clip_id = df.clip_id[index]
        if split == split_name:
            for j in range(1,11):
                feature_dir = 'features_norm/feature_%s_%d_%d.pickle' %(split,clip_id,j)
                features_dir.append(feature_dir)
                label = df.iloc[:,1:51][index:index+1].values.reshape(50).tolist()
                labels.append(label)
    df_features_dir = pd.DataFrame(features_dir,columns=['features_dir'])
    df_labels = pd.DataFrame(labels,columns=labels_column)
    return pd.concat([df_features_dir,df_labels],axis=1)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)

        if not os.path.exists('features_dir'):
            os.mkdir('features_dir')

        df_train_dir = create_features_norm_dir(df, 'train')
        df_train_dir.to_csv('features_dir/feature_train_dir.csv',index=None)

        df_val_dir = create_features_norm_dir(df, 'val')
        df_val_dir.to_csv('features_dir/feature_val_dir.csv',index=None)
    
    else:
        print('usage: python create_features_norm_dir.py input_path:annotations_final_top_50_tag.csv')
