"""A simple model using LSTM

Routine:
    1. Load MFCC with variable time length: [N, n_mfcc, ]
    2. lstm -> use last output [N, 256]
    4. Dense(1024)
    5. Dense(1024)
    6. Softmax(C)

Result: in fma dataset, got near 50 acc wihout any optim
    
"""

# pylint: disable=missing-docstring

import os
import sys
import json

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
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
        self.X, self.Y, self.TestX, self.TestY = fma.PrepareData(
            fma.FindFiles())
        self.BatchSize = 4
        self.dropout = 0.5
        self.n_count = 0
        self.n_mfcc = 32

    def GetNext16Batch(self, j):
        if self.n_count == 0:
            self.n_count = j * 16 * self.BatchSize
        batch_x, batch_y = fma.GetProcessedMatrix(self.X[self.n_count:(
            16*self.BatchSize + self.n_count)], self.Y[self.n_count:(
                16*self.BatchSize + self.n_count)])

        self.n_count += batch_x.shape[0]
        self.n_count %= len(self.X)
        remains = 16 * self.BatchSize - batch_x.shape[0]

        # some audio may broken, ensure finally get all values
        while batch_x.shape[0] < (16 * self.BatchSize):
            remain_x, remain_y = fma.GetProcessedMatrix(
                self.X[self.n_count:(self.n_count+remains)], self.Y[self.n_count:(self.n_count+remains)])
            batch_x = np.concatenate((batch_x, remain_x))
            self.n_count += batch_x.shape[0]
            self.n_count %= len(self.X)
            remains = 16 * self.BatchSize - batch_x.shape[0]
            batch_y = np.concatenate((batch_y, remain_y))
        y = np.zeros((batch_y.shape[0], 16))
        y[range(batch_y.shape[0]), batch_y] = 1
        return batch_x, y


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

    def Build(self, mode, epoch=12):
        # create model
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, stateful=True, input_shape=(None, self.n_mfcc),
                batch_input_shape=(16 * self.BatchSize, None, self.n_mfcc)))
        model.add(LSTM(256))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(16, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])


        batch_per_epoch = len(self.X) // self.BatchSize
        for i in range(epoch):
            for j in range((batch_per_epoch // 16)):
                with open(MODEL_PATH+'ijk.json', 'w') as fp:
                    json.dump({'i': i, 'j': j}, fp)
                batchX, batchY = self.GetNext16Batch(j)
                print("next batch: %d" %
                        j, batchX.shape, batchY.shape)
                l = model.train_on_batch(batchX, batchY)
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
