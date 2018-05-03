import os
import sys

import librosa as r
import numpy as np
import pandas as pd
from random import shuffle
from multiprocessing.dummy import Pool as ThreadPool

import json

LOG_LEVEL = 0

read_count = 0



DATASET_PATH = './data/fma_medium'

TRACK_CSV = './data/fma_metadata/tracks.csv'

FMA_DICT = './data/fma_genredict.json'


def FindFiles():
    if LOG_LEVEL < 1:
        print('FindFiles()')
    if os.path.exists(FMA_DICT):
        with open(FMA_DICT, 'r') as fp:
            return json.load(fp)
    if os.path.exists(DATASET_PATH):
        X = []
        for route in os.listdir(path=DATASET_PATH):
            dirpath = os.path.join(DATASET_PATH, route)
            files = [os.path.abspath(os.path.join(dirpath, f))
                     for f in os.listdir(dirpath)]
            X += files
        return GetGenreDict(X)
    else:
        raise FileNotFoundError('Please extract dataset first')


def Spectrogram(pcm, freq=512, t=128):
    if LOG_LEVEL < 1:
        print('spectrogram()')
    # input a [samples, ] vertor, output a [1 + n_fft/2, samples / hop], and n_fft = 2048, hop_length = 512 by default
    hop_length = np.math.floor(pcm.shape[0] / t) + 1
    # return [freq + 1, t]
    return r.stft(pcm, n_fft=(freq * 2), hop_length=hop_length)


def Read(pcm):
    if LOG_LEVEL < 1:
        print('Read()')
    return Spectrogram(pcm).T


def GetGenreDict(files):
    if LOG_LEVEL < 1:
        print('GetGenreDict()')
    # firstly, convert the sub-genres to the root genre (163 to 8 class)
    tracks = pd.read_csv(TRACK_CSV, header=[0, 1, 2])
    names = [int(os.path.splitext(os.path.basename(i))[0]) for i in files]

    mapping = dict()
    i = 0
    for n in names:
        genre = tracks.loc[tracks[('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'track_id')] == n].iloc[0][(
            'track', 'genre_top', 'Unnamed: 40_level_2')]
        if genre not in mapping:
            mapping[genre] = []
        mapping[genre].append(files[i])
        i += 1
    with open(FMA_DICT, 'w') as fp:
        json.dump(mapping, fp)
    return mapping


def PrepareData(d):
    ''' Notice that here d is a dict of the path list of all genres' all files '''
    matX = []
    matY = []
    i = 0
    for key, value in d.items():
        for f in value:
            matX.append(f)
            matY.append(i)
        i += 1
    concat = list(zip(matX, matY))
    shuffle(concat)
    x, y = zip(*concat)
    count = len(x)
    testCount = count // 10
    trainX = x[:9*testCount]
    trainY = y[:9*testCount]
    testX = x[9*testCount:]
    testY = y[:9*testCount]
    # X is a list of file paths
    # Y is a list of int labels, this label is the index of dict's key
    return trainX, trainY, testX, testY


def GetPCM(path):
    global read_count
    try:
        pcm, _ = r.load(path)
        if LOG_LEVEL < 1:
            print(read_count)
            read_count += 1
        return pcm
    except:
        # broken file
        return None


def GetProcessedMatrix(X, y):
    ''' Notice that here X is a list of the path list of a few files '''
    if LOG_LEVEL < 1:
        print('GetProcessedMatrix()')

    # make the Pool of workers
    pool = ThreadPool(8)
    # open the urls in their own threads
    # and return the results
    pcms = pool.map(GetPCM, X)

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # some audio files are broken, remove them
    y = [val for idx, val in enumerate(y) if pcms[idx] is not None]
    pcms = [x for x in pcms if x is not None]

    # make the Pool of workers
    pool = ThreadPool(8)
    # open the urls in their own threads
    # and return the results
    results = pool.map(Read, pcms)

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    return np.absolute(np.array(results))[:, :, :, None], np.array(y)
