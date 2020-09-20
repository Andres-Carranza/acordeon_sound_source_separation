import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import wavfile 
from tqdm import tqdm
from shutil import copyfile

def check_dir(save_dir):
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

def find_available(source,typ):
    global acordeon_path

    i = 0
    check_name = '{}{}.{}'.format(source, i, typ)

    while os.path.exists(os.path.join(acordeon_path,source,check_name)):
        i+=1
        check_name ='{}{}.{}'.format(source, i, typ)
    return check_name

def clean_data(sources, typ='mp4'):
    global acordeon_path

    for source in sources:
        src_path = os.path.join(acordeon_path,source)
        dest_path= os.path.join(acordeon_path, 'raw', typ)

        check_dir(dest_path)

        for i, fn in enumerate(sorted(os.listdir(src_path))):
            if fn == '.DS_Store':
                continue
            nfn = '{}{}.{}'.format(source, i, typ)
            #print(fn, nfn , end=' ')
            if fn == nfn:
                continue
            if os.path.exists(os.path.join(acordeon_path, source, nfn)):
                nfn = find_available(source,typ)
                #print(nfn , end=' ')
            os.rename(os.path.join(src_path,fn), os.path.join(src_path,nfn))
        
            copyfile(os.path.join(src_path,nfn), os.path.join(dest_path,nfn))
    
            #print()

def to_wav( file_type='mp4'):
    global acordeon_path

    check_dir(os.path.join(acordeon_path,'raw/wav'))
    clips = sorted(os.listdir(os.path.join(acordeon_path,'raw',file_type)))
    
    for clip in clips:
        if clip == '.DS_Store':
            continue
        name = clip.split('.')[0]
        clip_path = os.path.join(acordeon_path,'raw',file_type,'{}.{}'.format(name,file_type))
        wav_path = os.path.join(acordeon_path,'raw','wav',name+'.wav')
        if os.path.exists(wav_path) is False:
            os.system('ffmpeg -i {} -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav {}'\
                .format(clip_path, wav_path))

def split_data(source,dest='raw/split',time =5):
    global acordeon_path

    clips = sorted(os.listdir(os.path.join(acordeon_path,source)))

    if os.path.exists(os.path.join(acordeon_path,source,'.DS_Store')):
        clips.remove('.DS_Store')


    for  fn in clips:  
        interval = 44100 * time
            
        wav_path = os.path.join(acordeon_path,source,fn)
        rate, wav = wavfile.read(wav_path)
        
        stop = (wav.shape[0] // interval) * interval

        for i in range(0, stop-interval, interval):
            sample = wav[i:i+interval]

            save_dir = os.path.join(acordeon_path, dest)
            check_dir(save_dir)

            fn = fn.split('.')[0]

            i = int(i/interval)

            save_fn = str(os.path.join(save_dir,'{}_{}.wav'.format(fn,i)))

            if os.path.exists(save_fn):
                continue
            wavfile.write(filename=save_fn,
            rate = rate,
            data = sample)




def format_acordeon_data():
    global acordeon_path

    acordeons = sorted(os.listdir(os.path.join(acordeon_path,'raw/split')))

    if os.path.exists(os.path.join(acordeon_path,'raw/split/.DS_Store')):
        acordeons.remove('.DS_Store')


    train_fns, valid_fns = train_test_split(acordeons, test_size = 0.1,random_state=0)

    for split, fns in zip(['train', 'valid'], [train_fns, valid_fns]):
        for fn in tqdm(fns):
            wav_path = os.path.join(acordeon_path,'raw/split',fn)

            save_dir = os.path.join(acordeon_path, 'data', split, 'acordeon')
            check_dir(save_dir)

            save_fn = str(os.path.join(save_dir, fn))

            if os.path.exists(save_fn):
                continue

            copyfile(wav_path,save_fn)

def format_interferer_data():
    global acordeon_path

    clips = sorted(os.listdir(os.path.join(acordeon_path,'salsa/split')))

    if os.path.exists(os.path.join(acordeon_path,'salsa/split/.DS_Store')):
        clips.remove('.DS_Store')


    train_fns, valid_fns = train_test_split(clips, test_size = 0.1,random_state=0)

    for split, fns in zip(['train', 'valid'], [train_fns, valid_fns]):
        for fn in tqdm(fns):
            wav_path = os.path.join(acordeon_path,'salsa/split',fn)

            save_dir = os.path.join(acordeon_path, 'data', split, 'interferer')
            check_dir(save_dir)

            save_fn = str(os.path.join(save_dir, fn))

            if os.path.exists(save_fn):
                continue

            copyfile(wav_path,save_fn)


acordeon_path = "/Users/andres/Documents/acordeon"

sources = ['franfaleromusic', 'juanjosegranados','silveravallenato']
#clean_data(sources)
#to_wav('mp4')
#split_data('raw/wav')
#format_acordeon_data()
split_data('salsa/raw','salsa/split')

format_interferer_data()

'''def format_interferer_data():
    global acordeon_path

    df = pd.read_csv(os.path.join(acordeon_path, 'salsa', 'split', 'esc50.csv'))

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
                os.system('cp {} {}'.format(src_path, fn_path))'''