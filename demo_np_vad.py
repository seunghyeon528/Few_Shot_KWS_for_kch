import numpy as np
import pyaudio
import scipy.io.wavfile as wav
import threading
from multiprocessing import Process, Queue, Array
import pdb
import warnings
import argparse
import time
import os, sys
import subprocess
import pickle
import time
import os
import json
import math
from tqdm import tqdm
import torchaudio
import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import pdb


########################################################################################################################
#                                           THREAD function
########################################################################################################################
def few_shot_inference(input_chunk):
    # normalize
    input_chunk = input_chunk.astype(np.float32)
    input_chunk = input_chunk / (np.max(input_chunk) - np.min(input_chunk))

    # numpy to tensor, extract mfcc feature
    input_chunk = torch.tensor(input_chunk)
    xq = extract_features(input_chunk)  # [1. 51. 40] , [1 , T, f]
    xq = xq.unsqueeze(0)
    xq = xq.unsqueeze(0)  # [1,1,1,51,40]

    sample = {}
    sample['xs'] = xs
    sample['xq'] = xq

    output = model.loss(sample)
    return output

def generate_input_chunk(input_chunk_segment):
    for i in range(len(input_chunk_segment)):
        if i == 0:
            input_chunk = input_chunk_segment[i]
        else:
            input_chunk = np.concatenate([input_chunk, input_chunk_segment[i]], axis = 1)
    return input_chunk

def get_vad_flag(input_chunk):
    power = np.abs(input_chunk).mean()
    if power >= 1000:
        vad = True
    else:
        vad = False
    return vad

def start_KWS(Queue_audio):
    chunk_list = []
    while True:
        chunk = Queue_audio.get()
        chunk_list.append(chunk)

        if len(chunk_list) >=4:
            input_chunk = generate_input_chunk(chunk_list[:4]) # select last 4 chunk and concat them so as to make 1 sec input
            del chunk_list[0]

            vad = get_vad_flag(input_chunk)
            #print(vad)

            if vad:
                output = few_shot_inference(input_chunk)
            else:
                output = torch.tensor([4])
            print(output)


def start_stream(Queue_audio):
    while True:
        data = stream.read(CHUNK_SIZE)
        data = np.fromstring(data, dtype = np.int16)
        data = np.expand_dims(data, axis=0) # [1, 4000]

        Queue_audio.put(data)

########################################################################################################################
#                                           FEATURE EXTRACT function
########################################################################################################################
## hyperparameters used for extracting MFCC feature
window_size_ms = 40
window_stride_ms = 20
sample_rate = 16000
feature_bin_count = 40

def build_mfcc_extractor():
    frame_len = window_size_ms / 1000
    stride = window_stride_ms / 1000
    mfcc = torchaudio.transforms.MFCC(sample_rate,
                                    n_mfcc=feature_bin_count,
                                    melkwargs={
                                        'hop_length' : int(stride*sample_rate),
                                        'n_fft' : int(frame_len*sample_rate)})
    return mfcc

mfcc = build_mfcc_extractor()

def extract_features(sound):
    features = mfcc(sound)[0] # just one channel
    features = features.T # f x t -> t x f
    features = torch.unsqueeze(features,0)
    return features



########################################################################################################################
#                                           MAIN function
########################################################################################################################
def check_audio_env(pypy):
    print('============================================')
    print(pypy.get_device_count())
    print('============================================')
    for index in range(pypy.get_device_count()):
        desc = pypy.get_device_info_by_index(index)
        print("DEVICE: %s INDEX: %s RATE:%s"%(desc['name'],index,int(desc["defaultSampleRate"])))

if __name__ == '__main__':

    ## -- 1. load model
    model_path = "./results/best_model.pt"
    model = torch.load(model_path)
    model.eval()


    ## -- 2. load support data
    support_path = "./support.pt"
    xs = torch.load(support_path) # [3, 10, 1, 51, 40], [3-way, 10-shots, 1, T, f]


    ## -- 3. audio setting
    RATE  = 16000
    CHUNK_SIZE = int(16000/4) # 1 sec

    Queue_audio = Queue()
    pypy = pyaudio.PyAudio()

    check_audio_env(pypy)
    pdb.set_trace() # !! select input_device_index
    stream = pypy.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, input_device_index=2, frames_per_buffer=CHUNK_SIZE)


    ## -- 4. start thread
    streaming = threading.Thread(target=start_stream, args=(Queue_audio,))
    KWS_spotting = threading.Thread(target=start_KWS, args=(Queue_audio,))

    streaming.start()
    KWS_spotting.start()

    streaming.join()
    KWS_spotting.join()
