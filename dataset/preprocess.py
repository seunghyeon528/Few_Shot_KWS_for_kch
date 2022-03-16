################################################################################################
# change test wav as googlespeech command datset type (pcm16, sr = 16000, mono)
################################################################################################
import librosa
import pdb
import numpy as np
import os
from tqdm import tqdm
import soundfile as sf

## -- search list of wav files 
def search(dirname,end):
    global wav_list
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename,end)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == end: 
                    wav_list.append(full_filename)
    except PermissionError:
        pass

root_dir = "./original"
save_root_dir = "./preprocess"
wav_list = []
search(root_dir, ".wav")
pdb.set_trace()

## -- change each wav file format and resave
for wav_path in tqdm(wav_list):
    ## -- read
    audio_wav, _ = librosa.load(wav_path, sr = 16000)
    
    ## -- pad to 16000    
    origin_len = len(audio_wav)
    if origin_len <16000:
        start_num = np.random.randint(0,16000-origin_len-2)
        empty_wav = np.zeros([16000])
        empty_wav[start_num:start_num+audio_wav.shape[0]]=audio_wav
        audio_wav = empty_wav
    else:
        start_num = np.random.randint(0,origin_len-16000-2)
        #audio_wav = audio_wav[start_num:start_num+16000]
        audio_wav = audio_wav[-16000:]

    ## find save_path
    pdb.set_trace()
    label = wav_path.split("/")[-2]
    save_path = wav_path.replace(root_dir, save_root_dir)
    
    ## -- write
    sf.write(save_path, audio_wav, 16000, 'PCM_16')
    
## -- read
# data, _ = librosa.load(mp4_path, sr = 16000)
# pdb.set_trace()

# ## -- save as .wav format
# save_path = "/home/jungwook/lsh/AVSE_demo/preprocess/check_wav/Lsh_wav/00010_lsh_16.wav"

# import soundfile as sf
# sf.write(save_path, data, 16000,  'PCM_16')
