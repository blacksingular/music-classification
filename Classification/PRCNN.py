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

"""
Created by zxs
Revised by gy
"""

# pylint: disable=missing-docstring

import os
import sys

import numpy as np
import tensorflow as tf

import GTZAN as G

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
MODEL_PATH = './tfmodel'
class PRCNN():
    """Parelleling Bi-RNN + CNN"""

    def __init__(self):
        x, y = G.getPCMmatrix()
        self.RawX, self.Y, self.RawValX, self.ValidY, self.RawTestX, self.TestY = G.PrepareData(x, y)
        self.X = G.getProcessedMatrix(self.RawX)
        self.TestX = G.getProcessedMatrix(self.RawTestX)
        self.ValidX = G.getProcessedMatrix(self.RawValX)
        print('X: ', self.X.shape)
        print('Y: ', self.Y.shape)
        print('TestX: ', self.TestX.shape)
        print('TestY: ', self.TestY.shape)
        self.BatchSize = 30
        self.dropout = 0.5

    def batch_generator(self):
        batch_x = []
        batch_y = []
        shuffle_indices = np.random.permutation(np.arange(len(self.Y)))
        self.X = self.X[shuffle_indices]
        self.Y = self.Y[shuffle_indices]
        for i in range(self.X.shape[0]):
            batch_x.append(self.X[i])
            batch_y.append(self.Y[i])
            if len(batch_x) >= self.BatchSize:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

    def normalized_Inference_CNN(self, X, state):
        if state:
            X = tf.layers.batch_normalization(X, axis=-1, training=True)
            conv1 = tf.layers.conv2d(X, filters=16, kernel_size=(
                3, 1), strides=1, padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=True)
            conv1 = tf.nn.relu(conv1)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2)
            conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2)
            conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2)
            conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(conv4, pool_size=(4, 4), strides=4)
            conv5 = tf.layers.conv2d(pool4, filters=64, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(conv5, pool_size=(4, 4), strides=4)
        else:
            X = tf.layers.batch_normalization(X, axis=-1, training=False)
            conv1 = tf.layers.conv2d(X, filters=16, kernel_size=(
                3, 1), strides=1, padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=False)
            conv1 = tf.nn.relu(conv1)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2)
            conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2)
            conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(
               3, 1), strides=1, padding='same', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2)
            conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(conv4, pool_size=(4, 4), strides=4)
            conv5 = tf.layers.conv2d(pool4, filters=64, kernel_size=(
                3, 1), strides=1, padding='same', activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(conv5, pool_size=(4, 4), strides=4)

        flatten = tf.layers.flatten(out)
        print('CNN builded', flatten.shape)
        return flatten

    def Inference_CNN(self, X):
        conv1 = tf.layers.conv2d(X, filters=16, kernel_size=(
            3, 1), strides=1, padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2)
        conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=(
            3, 1), strides=1, padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2)
        conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(
           3, 1), strides=1, padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2)
        conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=(
            3, 1), strides=1, padding='same', activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(4, 4), strides=4)
        conv5 = tf.layers.conv2d(pool4, filters=64, kernel_size=(
            3, 1), strides=1, padding='same', activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(conv5, pool_size=(4, 4), strides=4)

        flatten = tf.layers.flatten(out)
        print('CNN builded', flatten.shape)
        return flatten

    def Inference_RNN_(self, X, mode=tf.estimator.ModeKeys.TRAIN):
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
    
    def Inference_RNN(self, X, mode=tf.estimator.ModeKeys.TRAIN):
        pool1 = tf.layers.max_pooling2d(X, pool_size=(1, 2), strides=(1, 2))
        pool1 = tf.reshape(pool1, (-1, 128, 256))
        print(pool1.get_shape())
        fw = tf.contrib.rnn.BasicLSTMCell(
            1, forget_bias=1.0, state_is_tuple=True)
        bw = tf.contrib.rnn.BasicLSTMCell(
            1, forget_bias=1.0, state_is_tuple=True)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, pool1, dtype=tf.float32, time_major=False)
        out = tf.concat(outputs, 1)
        out = tf.reshape(out, (-1, 256))
        print(out.shape)
        return out

    def Inference_Dense(self, cnn, lstm, mode=tf.estimator.ModeKeys.TRAIN):
        init = tf.concat([cnn, lstm], axis=1)
        dense = tf.layers.dense(cnn, units=10)
        print('Dense builded')
        return dense

    def Optimize(self, output, label):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(label, depth=10), logits=output))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer().minimize(loss)
        print('Optimize()')
        return loss, optimizer

    def Build(self, mode=tf.estimator.ModeKeys.TRAIN):
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

    def train(self, epochs=1000):
        if not self.builded:
            raise Exception()
        acc_old = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('load from checkpoint')
            # sess.graph.finalize()
            for epoch in range(epochs):
                step = 0
                batch = self.batch_generator()
                while step <= self.X.shape[0] / self.BatchSize:
                    try:
                        batchX, batchY = next(batch)
                    except StopIteration:
                        pass
                    # don't forget that batchY is one-hot label, convert it using tf.one_hot in loss
                    l, _ = sess.run([self.loss, self.optimizer], feed_dict={
                        self.varX: batchX, self.varLabel: batchY})
                    if step % 10 == 0:
                        print("epoch %d, step %d, loss: %f" % (epoch, step, l))
                    step += 1
                correct = self.Validation(sess, self.ValidX, self.ValidY)
                acc = correct[correct].size / correct.size
                print("acc: ", format(acc))
                if acc > acc_old:
                    acc_old = acc
                    self.saver.save(sess, MODEL_PATH + '/model.ckpt')
                    print('model saved')

    def test(self):
        if not self.builded:
            raise Exception()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('load from checkpoint')
                print("test: ")
                correct = self.Validation(sess, self.TestX, self.TestY)
                print("acc: %f" % (correct[correct].size / correct.size))
            else:
                print('no model exists')
