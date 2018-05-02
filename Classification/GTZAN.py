import os
import sys

import librosa as ros
import numpy as np
from random import shuffle

""" Download the GTZAN dataset, and extract into following path """


DATASET_PATH = './data/gtzan/genres'
READ_MATRIX_DUMP = './data/gtzan/loaded'


def getGenreDict():
    if os.path.exists(DATASET_PATH):
        dic = {}
        for route in os.listdir(path=DATASET_PATH):
            dirpath = os.path.join(DATASET_PATH, route)
            files = [os.path.abspath(os.path.join(dirpath, f))
                     for f in os.listdir(dirpath)]
            dic[route] = files
        return dic
    else:
        raise FileNotFoundError('Please extract dataset first')


def readFiles(dic):
    X = []
    y = []
    i = 0
    for key, value in dic.items():
        l = len(value)
        for idx, path in enumerate(value):
            # [n, ] and sample rate
            # because sr = k(samples/sec), so n = sr * duration
            # the amplitude of pcm is -1f ~ 1f
            pcm, sr = ros.load(path)
            X.append(pcm)
            y.append(i)
            print('%d of %d files read' % (idx, l))
            print("length: %d" % pcm.shape)
        print("%d of %d directory read" % (i+1, len(dic)))
        i += 1

    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    # returns X: [N, ] and each is a numpy array of the N's audio's sample time series, in GTZAN, N = 10000, class = 10
    #         y: [N, ] the label
    return X, y

# the count of samples is around 661794 (variable length)
# [N, 66XXXX]


def getPCMmatrix():
    if os.path.exists(READ_MATRIX_DUMP+'.x'):
        matX = np.load(READ_MATRIX_DUMP+'.x')
        maty = np.load(READ_MATRIX_DUMP+'.y')
        print('getPCMmatrix(): ', matX.shape, maty.shape)
        maty = maty.astype(np.int32)
        return matX, maty
    else:
        dic = getGenreDict()
        matX, maty = readFiles(dic)
        maty = maty.astype(np.int32)
        matX.dump(READ_MATRIX_DUMP+'.x')
        maty.dump(READ_MATRIX_DUMP+'.y')
        print('getPCMmatrix(): ', matX.shape, maty.shape)
        return matX, maty


def spectrogram(pcm, freq=512, t=128):
    # input a [samples, ] vertor, output a [1 + n_fft/2, samples / hop], and n_fft = 2048, hop_length = 512 by default
    hop_length = np.math.floor(pcm.shape[0] / t) + 1
    # return [freq + 1, t]
    return ros.stft(pcm, n_fft=(freq * 2), hop_length=hop_length)


""" TODO: should this be cached rather than PCM? PCM cotains more data but is too large
          should have a try """


def getProcessedMatrix(matX):
    X = []
    for arr in matX:
        s = spectrogram(arr).T
        X.append(s)

    # [N, 128, 513], prepared for PRCNN
    X = np.absolute(np.array(X))
    print('getProcessedMatrix(): ', X.shape)
    return X[:, :, :, None]


def PrepareData(X, y):
    # shuffle, split to train, valid and test

    # shuffle [X, y] in same order
    p = np.random.permutation(y.shape[0])
    X = X[p]
    y = y[p]

    # split (train + valid) : test = 9 : 1
    k = X.shape[0] // 10
    trainX = X[:9*k]
    testX = X[9*k:]
    trainY = y[:9*k]
    testY = y[9*k:]

    print('PrepareData()')

    return trainX, trainY, testX, testY
