"""A simple model using LSTM

Routine:
    1. Load MFCC with variable time length: [N, t, n_mfcc]
    2. lstm -> use last output [N, 256]
    4. Dense(1024)
    5. Dense(1024)
    6. Softmax(C)

Result: in fma dataset, got near 50 acc without any optim


Routine 2:
    fmcc and spectrogram all go to seperate lstm, and concatnate outputs into dense layers

Result: fma got near 50 acc with batch normalization
    
"""

# pylint: disable=missing-docstring

import os
import sys
import json

import numpy as np

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, BatchNormalization
from keras import backend as K
import cntk as C

import fma

# disable INFO and WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_PATH = './PRCNN_Model/'


class PRCNN():
    """Parelleling Bi-RNN + CNN"""

    def __init__(self):
        # notice that x is the list of path lists, y is genre id list
        self.X, self.Y, self.TestX, self.TestY = fma.PrepareClusterData(fma.GetEchoNestFeatures())
        self.BatchSize = 64
        self.dropout = 0.5
        self.n_count = 0
        self.n_mfcc = 32
        self.n_freq = 65

    def GetNext16Batch(self, j):
        if self.n_count == 0:
            self.n_count = j * self.BatchSize
        specX, mfccX, batch_y = fma.GetProcessedMatrix(self.X[self.n_count:(self.BatchSize + self.n_count)], self.Y[self.n_count:(self.BatchSize + self.n_count)])

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
        y = np.zeros((batch_y.shape[0], 4))
        y[range(batch_y.shape[0]), batch_y] = 1
        return specX, mfccX, y

    def Inference_RNN(self, X):
        pass
        # return results

    def Inference_Dense(self, lstm):
        pass
        # init = tf.concat([cnn, lstm], axis=1)
        # dense = tf.layers.dense(init, units=10)
        # print('Dense builded')
        # return dense

    def Optimize(self, output, label):
        pass
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.one_hot(label, depth=10), logits=output))
        # optimizer = tf.train.AdamOptimizer().minimize(loss)
        # print('Optimize()')
        # return loss, optimizer

    def Build(self, epoch=12):
        # create model
        spec = Input(shape=(None, self.n_freq), batch_shape=(self.BatchSize, None, self.n_freq))
        mfcc = Input(shape=(None, self.n_mfcc), batch_shape=(self.BatchSize, None, self.n_mfcc))

        line_1 = BatchNormalization()(spec)
        line_1 = LSTM(64, return_sequences=True, stateful=True)(line_1)
        line_1 = LSTM(64)(line_1)

        line_2 = BatchNormalization()(mfcc)
        line_2 = LSTM(64, return_sequences=True, stateful=True)(line_2)
        line_2 = LSTM(64)(line_2)

        combine = Concatenate(axis=1)([line_1, line_2])

        dense = BatchNormalization()(combine)
        dense_1 = Dense(256, activation='relu')(dense)
        dense_2 = Dense(256, activation='relu')(dense_1)
        output = Dense(4, activation='softmax')(dense_2)

        model = Model(inputs=[spec, mfcc], outputs=output)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        batch_per_epoch = len(self.X) // self.BatchSize
        for i in range(epoch):
            for j in range(batch_per_epoch):
                with open(MODEL_PATH+'ijk.json', 'w') as fp:
                    json.dump({'i': i, 'j': j}, fp)
                specX, mfccX, batchY = self.GetNext16Batch(j)
                print("next batch: %d" %
                      j, specX.shape, mfccX.shape, batchY.shape)
                l = model.train_on_batch([specX, mfccX], batchY)
                model.save('./saved/rnn.h5')  # creates a HDF5 file
                print("epoch %d, batch %d, loss: " % (i, j), l)

        # print("test: ")
        # score = model.evaluate(x_val, y_val, batch_size=32, verbose=1)
        # print("acc: %f" % (correct[correct].size / correct.size))

    def Validation(self, sess, X, y):
        pass
        # labels = tf.constant(y)
        # eval_ = tf.nn.in_top_k(self.networkOutput, labels, 1)
        # correct = sess.run(eval_, feed_dict={self.varX: X})
        # return correct

    def RestoreIJK(self):
        pass
        # if os.path.exists(MODEL_PATH+'ijk.json'):
        #     with open(MODEL_PATH+'ijk.json', 'r') as fp:
        #         d = json.load(fp)
        #         return d['i'], d['j']
        # else:
        #     return 0, 0

    def train(self, epoch=12):
        pass
        # with tf.Session() as sess:

        #     sess.run(tf.global_variables_initializer())
        #     self.saver = tf.train.Saver()
        #     ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         self.saver.restore(sess, ckpt.model_checkpoint_path)
        #         print('load from checkpoint')
        #         e, b = self.RestoreIJK()
        #         print(e, b)
        #     else:
        #         e = 0
        #         b = 0

        #     batch_per_epoch = len(self.X) // self.BatchSize

        #     for i in range(epoch - e):
        #         for j in range((batch_per_epoch // 16) - b):
        #             with open(MODEL_PATH+'ijk.json', 'w') as fp:
        #                 json.dump({'i': i, 'j': j}, fp)
        #             batchX, batchY = self.GetNext16Batch(j)
        #             print("next 16 batch: %d" %
        #                   (j*16), batchX.shape, batchY.shape)
        #             for k in range(16):
        #                 # don't forget that batchY is one-hot label, convert it using tf.one_hot in loss
        #                 l, _ = sess.run([self.loss, self.optimizer], feed_dict={
        #                     self.varX: batchX[self.BatchSize*k:self.BatchSize*(k+1)], self.varLabel: batchY[self.BatchSize*k:self.BatchSize*(k+1)]})
        #             print("epoch %d, batch %d, loss: %f" % (i, j*16 + k, l))
        #             self.saver.save(sess, MODEL_PATH + 'model.ckpt')

        #         b = 0
        #         correct = self.Validation(sess, self.TestX, self.TestY)
        #         print("acc: %f" % (correct[correct].size / correct.size))

        #     print("test: ")
        #     correct = self.Validation(sess, self.TestX, self.TestY)
        #     print("acc: %f" % (correct[correct].size / correct.size))
