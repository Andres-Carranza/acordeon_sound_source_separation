import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import wavfile 
from tqdm import tqdm

def check_dir(save_dir):
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)


acordeon_path = "/Users/andres/Documents/acordeon"

clips = os.listdir(os.path.join(acordeon_path,'raw','mp4'))

for clip in clips:
    name = clip.split('.')[0]
    clip_path = os.path.join(acordeon_path,'raw','mp3',name+'.mp3')
    wav_path = os.path.join(acordeon_path,'raw','wav',name+'.wav')
    if os.path.exists(wav_path) is False:
        os.system('ffmpeg -i {} -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav {}'\
            .format(clip_path, wav_path))


acordeons = sorted(os.listdir(os.path.join(acordeon_path,'raw','wav')))

train_path = os.path.join(acordeon_path, 'data', 'train','acordeon')
valid_path = os.path.join(acordeon_path, 'data', 'valid','acordeon')

train_fns, valid_fns = train_test_split(acordeons, test_size = 0.2)

for split, fns in zip(['train', 'valid'], [train_fns, valid_fns]):
    interval = 44100 * 5
    for fn in tqdm(fns):
        wav_path = os.path.join(acordeon_path,'raw','wav',fn)
        rate, wav = wavfile.read(wav_path)
        
        stop = (wav.shape[0] // interval) * interval

        for i in range(0, stop-interval, interval):
            sample = wav[i:i+interval]

            save_dir = os.path.join(acordeon_path, 'data', split, 'acordeon')
            check_dir(save_dir)

            fn = fn.split('.')[0]

            i = int(i/interval)

            save_fn = str(os.path.join(save_dir, fn+'_{}.wav'.format(i)))

            if os.path.exists(save_fn):
                continue
            wavfile.write(filename=save_fn,
            rate = rate,
            data = sample)


df = pd.read_csv(os.path.join(acordeon_path, 'ESC-50', 'meta', 'esc50.csv'))

inter_dirs = np.unique(df.category).tolist()

for cat in inter_dirs:
    df_cat = df[df.category==cat]

    train_df, valid_df = train_test_split(df_cat, test_size = 0.1)

    for split, split_df in zip(['train', 'valid'], [train_df, valid_df]):
        dir_path = os.path.join(acordeon_path, 'data', split,'interferer')

        check_dir(dir_path)

        for i, row in split_df.iterrows():
            fn = row['filename']
            fn_path = os.path.join(dir_path, fn)

            if os.path.exists(fn_path):
                continue
            src_path = os.path.join(acordeon_path, 'ESC-50', 'audio', fn)
            os.system('cp {} {}'.format(src_path, fn_path))



