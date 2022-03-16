import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import pdb

import torchaudio

## hyperparameters used for extracting MFCC feature
window_size_ms = 40
window_stride_ms = 20
sample_rate = 16000
feature_bin_count = 40

def load_audio(file_path):
    sound, _ = torchaudio.load(filepath=file_path, normalize=True,
                                        num_frames=16000)
    return sound

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

if __name__ == '__main__':
    
    ## -- 1. load model
    model_path = "./results/best_model.pt"
    model = torch.load(model_path)
    model.eval()

    ## -- 2. load support data
    support_path = "./support.pt"
    xs = torch.load(support_path) # [3, 10, 1, 51, 40], [3-way, 10-shots, 1, T, f]


    #################################################################
    #           BELOW HERE, CODE SHOULD BE PUT INTO THREAD
    #################################################################

    ## -- 3. load query data from wav file
    query_path = "/home/lsh/KWS/FS_KWS_LSH/dataset/preprocess/1/1_kch.wav"
    sound = load_audio(query_path)
    xq = extract_features(sound) # [1. 51. 40] , [1 , T, f]
    xq = xq.unsqueeze(0)
    xq = xq.unsqueeze(0) # [1,1,1,51,40]

    ## -- 4. inference
    sample = {}
    sample['xs'] = xs
    sample['xq'] = xq

    output = model.loss(sample)
    print(query_path)
    print(output) 

    ####################### NOTE ########################
    # model class path = "./protonets/models/few_shot.py"
    # output label : 0 - jump , 1 - dduiyeo, 2 - suggyeo
    #####################################################