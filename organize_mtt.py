import sys
import pandas as pd

# 使用頻度が高い50個のタグに絞り込む
def delete_low_frequency_tags(df_input):
    # タグのヒストグラムは https://github.com/keunwoochoi/magnatagatune-list にまとまっている）
    top50 = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 
            'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 
            'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 
            'weird', 'country', 'metal', 'female voice', 'choral']

    df_top50 = df_input[top50 + ['clip_id', 'mp3_path']]

    # タグが割振られていないサンプルを削除する（25863曲 => 21111曲）
    df_top50 = df_top50[df[top50].sum(axis=1) != 0]

    return df_top50

# train, validation, test（15250曲, 1529曲, 4332曲）に分割
def split_dataset(df_input):
    split_list = []
    for i in df_input['mp3_path']:
        if i[0] in ['0','1','2','3','4','5','6','7','8','9','a','b']:
            split = 'train'
        elif i[0] in ['c']:
            split = 'val'
        elif i[0] in ['d','e','f']:
            split = 'test'
        split_list.append(split)

    df_input['split'] = split_list
    df_split = df_input

    return df_split

if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]

        df = pd.read_csv(input_path, delimiter='\t')
        df_top50 = delete_low_frequency_tags(df)
        df_top50_split = split_dataset(df_top50)
        df_top50_split.to_csv(output_path)
        
        print('create %s!' %output_path)
    else:
        print('usage: python organize_mtt.py input_path:annotations_final.csv output_path:annotations_final_top_50_tag.csv')