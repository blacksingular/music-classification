# pylint: disable=missing-docstring

import os
import sys
import json

import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, BatchNormalization
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score, KFold

import cntk as C

import fma

# disable INFO and WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_PATH = './Regression_Model/'


class Regression():
    def __init__(self):
        # notice that x is the list of path lists, y is genre id list
        self.X, self.Y, self.TestX, self.TestY = fma.PrepareRegressionData(
            fma.GetEchoNestFeatures())
        self.BatchSize = 128
        self.dropout = 0.5
        self.n_count = 0
        self.n_mfcc = 32
        self.n_freq = 65

    def GetNextBatch(self, j):
        if self.n_count == 0:
            self.n_count = j * self.BatchSize
        specX, mfccX, batch_y = fma.GetProcessedMatrix(self.X[self.n_count:(
            self.BatchSize + self.n_count)], self.Y[self.n_count:(self.BatchSize + self.n_count)])

        self.n_count += specX.shape[0]
        self.n_count %= len(self.X)
        remains = self.BatchSize - specX.shape[0]

        # some audio may broken, ensure finally get all values
        while specX.shape[0] < (self.BatchSize):
            remain_spec_x, remain_mfcc_x, remain_y = fma.GetProcessedMatrix(
                self.X[self.n_count:(self.n_count+remains)], self.Y[self.n_count:(self.n_count+remains)])
            specX = np.concatenate((specX, remain_spec_x))
            mfccX = np.concatenate((mfccX, remain_mfcc_x))
            self.n_count += specX.shape[0]
            self.n_count %= len(self.X)
            remains = self.BatchSize - specX.shape[0]
            batch_y = np.concatenate((batch_y, remain_y))
        return specX, mfccX, batch_y



    def CreateModel(self):
        # create model
        spec = Input(shape=(None, self.n_freq), batch_shape=(8, None, self.n_freq))
        # mfcc = Input(shape=(None, self.n_mfcc), batch_shape=(self.BatchSize, None, self.n_mfcc))

        line_1 = BatchNormalization()(spec)
        line_1 = LSTM(64, return_sequences=True, stateful=True)(line_1)
        line_1 = LSTM(64)(line_1)

        # line_2 = BatchNormalization()(mfcc)
        # line_2 = LSTM(64, return_sequences=True, stateful=True)(line_2)
        # line_2 = LSTM(64)(line_2)

        # combine = Concatenate(axis=1)([line_1, line_2])

        dense = BatchNormalization()(line_1)
        dense_1 = Dense(256, activation='relu')(dense)
        dense_2 = Dense(256, activation='relu')(dense_1)
        output = Dense(6, activation='softmax')(dense_2)

        model = Model(inputs=spec, outputs=output)
        
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


    def Build(self, epoch=12):
        estimator = KerasRegressor(build_fn=self.CreateModel, nb_epoch=100, batch_size=8, verbose=0)
        batch_per_epoch = len(self.X) // self.BatchSize
        for i in range(epoch):
            for j in range(batch_per_epoch):
                # with open(MODEL_PATH+'ijk.json', 'w') as fp:
                #     json.dump({'i': i, 'j': j}, fp)
                specX, mfccX, batchY = self.GetNextBatch(j)
                print("next batch: %d" %
                      j, specX.shape, mfccX.shape, batchY.shape)
                results = cross_val_score(estimator, specX, batchY, n_jobs=-1)
                # model.save('./saved/rnn.h5')  # creates a HDF5 file
                print("epoch %d, batch %d, loss: " % (i, j), results)