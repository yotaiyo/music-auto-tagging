import pandas as pd
import random
import os
import pickle

def create_features_norm_dir(split_name, df):
    features_dir = []
    labels = []
    labels_column = df.iloc[:,1:51].columns.values.tolist()
    for index in df.index:
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
    df = pd.read_csv('annotations_final_top_50_tag.csv')

    if not os.path.exists('features_dir'):
        os.mkdir('features_dir')

    df_train_dir = create_features_norm_dir('train', df)
    df_train_dir.to_csv('features_dir/feature_train_dir.csv',index=None)

    df_val_dir = create_features_norm_dir('val', df)
    df_val_dir.to_csv('features_dir/feature_val_dir.csv',index=None)
