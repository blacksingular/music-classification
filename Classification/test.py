import os
import sys
import json
import librosa

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding
from keras import backend as K
import cntk as C

import fma


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def IthKey(keys, i):
    j = 0
    for k in keys:
        if j == i:
            return k
        j += 1


def OutputtoName(ids):
    ids = np.argmax(ids, axis=1)
    print(ids)
    with open('./data/fma_genredict.json', 'r') as fp:
        genredict = json.load(fp)
    keys = genredict.keys()
    return [IthKey(keys, i) for i in ids]


def LoadModel():
    model = load_model('./saved/rnn.h5')
    return model


def Predict(model, X):
    X = np.tile(X, (64, 1, 1))
    value = model.predict(X, batch_size=64)
    return OutputtoName(value)


def Eval(filePath):
    pcm, _ = librosa.load(filePath)
    print(pcm.shape)
    mfcc = librosa.feature.mfcc(y=pcm, n_mfcc=32)
    print(mfcc.shape)
    model = LoadModel()
    p = Predict(model, mfcc.T[np.newaxis, :, :])
    print(p)
