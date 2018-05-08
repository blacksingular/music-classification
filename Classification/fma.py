import os
import sys

import librosa as r
import numpy as np
import pandas as pd
from random import shuffle
from multiprocessing.dummy import Pool as ThreadPool
import uuid

from sklearn.cluster import KMeans

import json

LOG_LEVEL = 1


read_count = 0


DATASET_PATH = './data/fma_medium'

TRACK_CSV = './data/fma_metadata/tracks.csv'
ECHO_CSV = './data/fma_metadata/echonest.csv'

FMA_DICT = './data/fma_genredict.json'

ECHO_DICT = './data/fma_echo.json'

PRE_READ_PATH = './pre_read'


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


def AllFiles():
    if os.path.exists(DATASET_PATH):
        X = []
        for route in os.listdir(path=DATASET_PATH):
            dirpath = os.path.join(DATASET_PATH, route)
            files = [os.path.abspath(os.path.join(dirpath, f))
                     for f in os.listdir(dirpath)]
            X += files
        return X
    else:
        raise FileNotFoundError('Please extract dataset first')


def PrepareRegressionData(d):
    ''' Notice that here d is a dict of the path list of all genres' all files '''
    matX = []
    matY = []

    for key, value in d.items():
        matX.append(key)
        matY.append(value)

    concat = list(zip(matX, matY))
    shuffle(concat)
    x, y = zip(*concat)
    count = len(x)
    testCount = count // 10
    trainX = x[:9*testCount]
    trainY = y[:9*testCount]
    testX = x[9*testCount:]
    testY = y[9*testCount:]
    # X is a list of file paths
    # Y is a list of echo features(6 value per row)
    return trainX, trainY, testX, testY
    
def PrepareClusterData(d):
    ''' Notice that here d is a dict of the path list of all genres' all files '''
    matX = []
    matY = []

    for key, value in d.items():
        matX.append(key)
        matY.append(value)

    matY = np.array(matY)
    kmeans = KMeans(n_clusters=4, max_iter=5120, n_jobs=1).fit(matY)
    matY = kmeans.labels_.tolist()

    concat = list(zip(matX, matY))
    shuffle(concat)
    x, y = zip(*concat)
    count = len(x)
    testCount = count // 10
    trainX = x[:9*testCount]
    trainY = y[:9*testCount]
    testX = x[9*testCount:]
    testY = y[9*testCount:]
    # X is a list of file paths
    # Y is a list of echo features(6 value per row)
    return trainX, trainY, testX, testY


def GetEchoNestFeatures():
    if LOG_LEVEL < 1:
        print('GetEchoNestFeatures()')

    if os.path.exists(ECHO_DICT):
        with open(ECHO_DICT, 'r') as fp:
            return json.load(fp)

    files = AllFiles()
    # firstly, convert the sub-genres to the root genre (163 to 8 class)
    tracks = pd.read_csv(ECHO_CSV, header=[0, 1, 2, 3])
    names = [int(os.path.splitext(os.path.basename(i))[0]) for i in files]

    mapping = dict()
    i = 0
    for n in names:
        substi = tracks.loc[tracks[(
            'Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'Unnamed: 0_level_2', 'track_id')] == n]
        if len(substi) > 0:
            echos = (substi.iloc[0][('echonest', 'audio_features', 'acousticness')],
                     substi.iloc[0][(
                         'echonest', 'audio_features', 'danceability')],
                     substi.iloc[0][('echonest', 'audio_features', 'energy')],
                     substi.iloc[0][(
                         'echonest', 'audio_features', 'instrumentalness')],
                     substi.iloc[0][(
                         'echonest', 'audio_features', 'liveness')],
                     substi.iloc[0][('echonest', 'audio_features', 'speechiness')])
            mapping[files[i]] = [e.values[0] for e in echos]
        else:
            pass
        i += 1
    with open(ECHO_DICT, 'w') as fp:
        json.dump(mapping, fp)
    return mapping


def Spectrogram(pcm, freq=64, use_fixed_time=False):
    if LOG_LEVEL < 1:
        print('spectrogram()')
    # input a [samples, ] vertor, output a [1 + n_fft/2, samples / hop], and n_fft = 2048, hop_length = 512 by default
    # return [freq + 1, t]
    if pcm.shape[0] > 30 * 22050:
        pcm = pcm[:30 * 22050]
    else:
        pcm = np.pad(pcm, (0, 30 * 22050 - pcm.shape[0]), mode='constant')
    fft = r.stft(pcm, n_fft=(freq * 2), hop_length=512)
    return np.absolute(fft)


def MFCC(pcm):
    # cut or pad input
    if pcm.shape[0] > 30 * 22050:
        pcm = pcm[:30 * 22050]
    else:
        pcm = np.pad(pcm, (0, 30 * 22050 - pcm.shape[0]), mode='constant')
    return r.feature.mfcc(y=pcm, n_mfcc=32)


def Read(pcm):
    if LOG_LEVEL < 1:
        print('Read()')
    # [t, n_mfcc], t is (n_samples / hop)
    return Spectrogram(pcm).T, MFCC(pcm).T


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


def PreRead(mapping):
    if not os.path.exists(PRE_READ_PATH):
        os.makedirs(PRE_READ_PATH)
    file_dic = dict()
    for key, value in mapping.items():
        for path in value:
            try:
                pcm, _ = r.load(path)
                name = PRE_READ_PATH + '/' + str(uuid.uuid1())
                pcm.dump(name)
                if key not in file_dic:
                    file_dic[key] = []
                file_dic[key].append(name)
                print(key, name)
            except:
                pass
    with open(PRE_READ_PATH+'files.json', 'r') as fp:
        json.dump(file_dic, fp)
    return file_dic


def PrepareData(d):
    ''' Notice that here d is a dict of the path list of all genres' all files '''
    matX = []
    matY = []
    i = 0
    for _, value in d.items():
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

        # pcm = [time_series, ]
        # time_series is n samples
        # with sample rate = sr, we can find the duration is (n / sr) secs.
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

    return np.array([r[0] for r in results]), np.array([r[1] for r in results]), np.array(y)
