# music-auto-tagging

## Overview 
MagnaTagATune(MTAT)データセットを用いて、楽曲タグ予測モデルの構築から評価まで行うレポジトリ。

## データセットの準備
1, [ここ](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)からデータセットをダウンロード。
```
 annotations_final.csv
 mp3.zip.001
 mp3.zip.002
 mp3.zip.003
```
を取得。

2, 解凍して一つのファイルにまとめる。
```   
cat mp3.zip.* > mp3_all.zip  
unzip mp3_all.zip
```

3, python organize_mtt.py annotations_final.csv annotations_final_top_50_tag.csv  
MTATは25863曲、188個のタグから成るが、タグのヒストグラムがアンバランスであるため、使用頻度の高い上位50個のタグをしようすることが望ましい。（https://github.com/keunwoochoi/magnatagatune-list）  
そこで、余計なタグの削除とその結果、割り振られなくなったサンプルの削除を行う必要がある。

## mp3ファイルの分割
MTATのサンプルは30秒程度の長さからなるが、このままでは、モデルのインプットデータとして長すぎる。各サンプルを10分割して、3秒程度の長さをインプットデータとして使用するのが一般的である。

- python split_mp3.py annotations_final_top_50_tag.csv 22050 29 10

## 特徴量の抽出
[MIRtoolbox](https://www.jyu.fi/hytk/fi/laitokset/mutku/en/research/materials/mirtoolbox)を用い、13種類の特徴量を抽出する。  
計算性能に余力がある場合、wav_to_features.mの10行目を変更して、並列で計算を走らせると早く終わる。(ex. 1-7000,7001-14000,14001-21110に分割)

- wav_to_features(10, 22050, 0.050, 0.025)

## 正規化
1, 各特徴量の平均値と標準偏差を計算。 
- python save_statistics_of_features.py annotations_final_top_50_tag.csv  

2, 正規化
- python normalization.py annotations_final_top_50_tag.csv　　

計算性能に余力がある場合、normalization.pyの8行目を変更して、並列で計算を走らせると早く終わる。(ex. 0-5000,5000-10000,10000-15000,15000-21111に分割)

3, パスの作成
- python create_features_norm_dir.py annotations_final_top_50_tag.csv　　

## モデルのトレーニング
- python train.py annotations_final_top_50_tag.csv 32 3 0.001 0.5 0.01 23 100

## 作成したモデルの評価
-  python eval.py annotations_final_top_50_tag.csv 10 model/my_model_epoch-val_loss.hdf5  

評価指標として、50タグからroc-aucの平均値を計算する。0-1の範囲で大きいほど、予測性能の良いモデルであるといえる。



