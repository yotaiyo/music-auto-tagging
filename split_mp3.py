import sys
import wave
import struct
import math
import os
from scipy import frombuffer, int16
import pandas as pd

# mp3ファイルを均等に分割 + wavファイルへ変換
def split_mp3_to_wav_segment(filename, sampling_rate, mp3_length, num_segment, index):  
    wavf = str(filename) + '.wav'
    wr = wave.open(wavf, 'r')

    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate()
    fn = wr.getnframes()
    total_time = 1.0 * fn / fr
    t = mp3_length / num_segment 
    frames = int(ch * fr * t)

    data = wr.readframes(wr.getnframes())
    wr.close()
    X = frombuffer(data, dtype=int16)

    if total_time < mp3_length:
        print('mp3_length must be shorter than total_time')
        
    for j in range(num_segment):
        split = df.split[index]
        clip_id = df.clip_id[index]
        outf = 'wav_segment/'+ split + '_' + str(clip_id) + '_' + str(j+1) +'.wav'
        x = math.floor(sampling_rate*(total_time-mp3_length)/2) 
        start_cut = j*frames + x 
        end_cut = j*frames + frames + x
        Y = X[start_cut:end_cut]
        outd = struct.pack("h" * len(Y), *Y)
        ww = wave.open(outf, 'w')
        ww.setnchannels(ch)
        ww.setsampwidth(width)
        ww.setframerate(fr)
        ww.writeframes(outd)
        ww.close()

def create_wav_segment(df_input, sampling_rate, mp3_length, num_segment):
    for index in df_input.index:
        print('index', index)
        df_mp3_path = df_input.mp3_path[index]
        mp3_path = 'mp3/'+'%s' %df_mp3_path
        os.system("lame --resample 22.050 -b 32 -a '%s' resample.mp3" %mp3_path)
        os.system("lame --decode resample.mp3 'resample.wav'" )    
        file_name = 'resample'
        split_mp3_to_wav_segment(file_name, sampling_rate, mp3_length, num_segment ,index)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)

        sampling_rate = int(sys.argv[2])
        mp3_length = int(sys.argv[3])
        num_segment = int(sys.argv[4])

        if not os.path.exists('wav_segment'):
            os.mkdir('wav_segment')

        create_wav_segment(df, sampling_rate, mp3_length, num_segment)

        os.remove('resample.mp3')
        os.remove('resample.wav')
    else:
        print('usage: python split_mp3.py input_path:annotations_final_top_50_tag.csv sampling_rate:22050 mp3_length:29 num_segment:10')