import os
import pandas as pd

df = pd.read_csv('annotations_final_top_50_tag.csv')
ans = []
for index in df.index:
    split = df.split[index]
    clip_id = df.clip_id[index]
    path_exist = os.path.exists('features/zerocross_%s_%d_1.txt' %(split, clip_id))
    if not path_exist:
        ans.append(index)
if len(ans) > 0:
    print(ans[0])
    print(ans[len(ans)-1])
    print(len(ans))
else:
    print('OK!')