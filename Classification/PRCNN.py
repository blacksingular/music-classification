"""Parelleling Bi-RNN + CNN according to paper 1712.08370

Routine:
    1. Load STFT spectrum from GTZAN: [N, 128, 513]
    2.
        Path A:
            cnn(16, 3, 3, 1, 1) - pooling - cnn(32, 3, 3, 1, 1) - 
            pooling - cnn(64...) - pooling - cnn(128...) - pooling - 
            cnn(256...) - pooling - flattened output [N, 256]
        Path B:
            pooling(1, 2) strides(1, 2) to [N, 128, 256] - 
            embedding to [N, 128, 128] - BGRU-RNN - output [N, 256]
    3. Concate output A and B [N, 512]
    4. Dense(10)
    5. Softmax(10)


Result:
    When using GTZAN, this dataset is too small, and easily overfit, fin acc = 0.59 without any optimization

"""

# pylint: disable=missing-docstring

import os
import sys

import numpy as np
import tensorflow as tf

import GTZAN as G


class PRCNN():
    """Parelleling Bi-RNN + CNN"""

    def __init__(self):
        x, y = G.getPCMmatrix()
        self.RawX, self.Y, self.RawTestX, self.TestY = G.PrepareData(x, y)
        self.X = G.getProcessedMatrix(self.RawX)
        self.TestX = G.getProcessedMatrix(self.RawTestX)
        print('X: ', self.X.shape)
        print('Y: ', self.Y.shape)
        print('TestX: ', self.TestX.shape)
        print('TestY: ', self.TestY.shape)
        self.BatchSize = 32
        self.dropout = 0.5

    def GetNextBatch(self):
        index = np.random.choice(self.Y.shape[0], self.BatchSize)
        return self.X[index], self.Y[index]

    def Inference_CNN(self, X):
        conv1 = tf.layers.conv2d(X, filters=16, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2)
        conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2)
        conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2)
        conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=2)
        conv5 = tf.layers.conv2d(pool4, filters=256, kernel_size=(
            5, 5), strides=1, activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(conv5, pool_size=(2, 2), strides=2)

        flatten = tf.layers.flatten(out)
        print('CNN builded', flatten.shape)
        return flatten

    def Inference_RNN(self, X, mode=tf.estimator.ModeKeys.TRAIN):
        X = tf.reshape(X, (-1, 128, 513))
        shape = tf.shape(X)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            256, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(
            shape[0], dtype=tf.float32)  # 初始化全零 state
        outputs, _ = tf.nn.dynamic_rnn(
            lstm_cell, X, initial_state=init_state, time_major=False)
        print('RNN builded', outputs.shape)
        # 把 outputs 变成 列表 [(batch, outputs)..] * steps
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = outputs[-1]
        print(results.shape)
        return results

    def Inference_Dense(self, cnn, lstm, mode=tf.estimator.ModeKeys.TRAIN):
        init = tf.concat([cnn, lstm], axis=1)
        dense = tf.layers.dense(init, units=10)
        print('Dense builded')
        return dense

    def Optimize(self, output, label):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(label, depth=10), logits=output))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        print('Optimize()')
        return loss, optimizer

    def Build(self, mode):
        self.varX = tf.placeholder(dtype=tf.float32, shape=(None, 128, 513, 1))
        self.varLabel = tf.placeholder(dtype=tf.int32, shape=(None,))
        cnn = self.Inference_CNN(self.varX)
        lstm = self.Inference_RNN(self.varX, mode=mode)
        self.networkOutput = self.Inference_Dense(cnn, lstm, mode=mode)
        self.loss, self.optimizer = self.Optimize(
            self.networkOutput, self.varLabel)
        self.builded = True

    def Validation(self, sess, X, y):
        labels = tf.constant(y)
        eval_ = tf.nn.in_top_k(self.networkOutput, labels, 1)
        correct = sess.run(eval_, feed_dict={self.varX: X})
        return correct

    def train(self, epoch=3, batch_per_epoch=200):
        if not self.builded:
            raise Exception()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                for j in range(batch_per_epoch):
                    batchX, batchY = self.GetNextBatch()
                    # don't forget that batchY is one-hot label, convert it using tf.one_hot in loss
                    l, _ = sess.run([self.loss, self.optimizer], feed_dict={
                        self.varX: batchX, self.varLabel: batchY})
                    if j % 50 == 0:
                        print("epoch %d, batch %d, loss: %f" % (i, j, l))
                
                correct = self.Validation(sess, self.TestX, self.TestY)
                print("acc: %f" % (correct[correct].size / correct.size))

            print("test: ")
            correct = self.Validation(sess, self.TestX, self.TestY)
            print("acc: %f" % (correct[correct].size / correct.size))
