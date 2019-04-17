
function wav_to_features(num_segment, sampling_rate, hop_length, window_length)
% 分割されたwavファイルから13種類の特徴量を抽出する。
% sampling_rate = 22050[Hz], hop_length = 0.050, window_length=0.025が望ましい。

T = readtable('annotations_final_top_50_tag.csv','Delimiter',',');
mkdir features;
H = height(T);

for i = 1:H
    clip_id = T.clip_id(i);
    split = cell2mat(T.split(i));
    for j = 1:num_segment
        s = 'wav_segment/%s_%d_%d.wav';
        wav_path = sprintf(s,split,clip_id,j);
        
        wav_segment = miraudio(wav_path,'Sampling',sampling_rate);
        wav_segment_frame = mirframe(wav_segment,hop_length,'s',window_length,'s');
        wav_segment_spec = mirspectrum(wav_segment_frame);
        
        zerocross = mirzerocross(wav_segment_frame);
        rms = mirrms(wav_segment_frame);
        pulseclarity = mirpulseclarity(wav_segment,'Frame',1.0,'s',0.1,'s');
        beatspectrum = mirbeatspectrum(wav_segment);
        
        flux = mirflux(wav_segment_spec);
        centroid = mircentroid(wav_segment_spec);
        rolloff = mirrolloff(wav_segment_spec);
        flatness = mirflatness(wav_segment_spec);
        entropy = mirentropy(wav_segment_spec);
        skewness = mirskewness(wav_segment_spec);
        kurtosis = mirkurtosis(wav_segment_spec);
        chromagram = mirchromagram(wav_segment,'Frame');
        
        mfcc = mirmfcc(wav_segment_spec,'Rank',1:23);

        
        zerocross_path = sprintf('features/zerocross_%s_%d_%d.txt',split,clip_id,j);
        rms_path = sprintf('features/rms_%s_%d_%d.txt',split,clip_id,j);
        pulseclarity_path = sprintf('features/pulseclarity_%s_%d_%d.txt',split,clip_id,j);
        beatspectrum_path = sprintf('features/beatspectrum_%s_%d_%d.txt',split,clip_id,j);
        flux_path = sprintf('features/flux_%s_%d_%d.txt',split,clip_id,j);
        centroid_path = sprintf('features/centroid_%s_%d_%d.txt',split,clip_id,j);
        rolloff_path = sprintf('features/rolloff_%s_%d_%d.txt',split,clip_id,j);
        flatness_path = sprintf('features/flatness_%s_%d_%d.txt',split,clip_id,j);
        entropy_path = sprintf('features/entropy_%s_%d_%d.txt',split,clip_id,j);
        skewness_path = sprintf('features/skewness_%s_%d_%d.txt',split,clip_id,j);
        kurtosis_path = sprintf('features/kurtosis_%s_%d_%d.txt',split,clip_id,j);
        chromagram_path = sprintf('features/chromagram_%s_%d_%d.txt',split,clip_id,j);
        mfcc_path = sprintf('features/mfcc_%s_%d_%d.txt',split,clip_id,j);
        
        mirexport(zerocross_path,zerocross,'RAW');
        mirexport(rms_path,rms,'RAW');
        mirexport(pulseclarity_path,pulseclarity,'RAW');
        mirexport(beatspectrum_path,beatspectrum,'RAW');
        mirexport(flux_path,flux,'RAW');
        mirexport(centroid_path,centroid,'RAW');
        mirexport(rolloff_path,rolloff,'RAW');
        mirexport(flatness_path,flatness,'RAW');
        mirexport(entropy_path,entropy,'RAW');
        mirexport(skewness_path,skewness,'RAW');
        mirexport(kurtosis_path,kurtosis,'RAW');
        mirexport(chromagram_path,chromagram,'RAW');
        mirexport(mfcc_path,mfcc,'RAW');
    end
end
end