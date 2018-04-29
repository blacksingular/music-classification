import os
import sys

import librosa as ros
import numpy as np

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


def getPCMmatrix():
    if os.path.exists(READ_MATRIX_DUMP+'.x'):
        matX = np.load(READ_MATRIX_DUMP+'.x')
        maty = np.load(READ_MATRIX_DUMP+'.y')
        return matX, maty
    else:
        dic = getGenreDict()
        matX, maty = readFiles(dic)
        matX.dump(READ_MATRIX_DUMP+'.x')
        maty.dump(READ_MATRIX_DUMP+'.y')
        return matX, maty


def spectrogram(pcm):
    # input a [t, ] vertor, output a [1 + n_fft/2, t], n_fft = 2048 by default
    return ros.stft(pcm)
