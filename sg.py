from __future__ import print_function

import tensorflow as tf
import argparse
import os
import random
import imageio
import sys
import numpy as np
import shutil
import Evaluation_script_
from PIL import Image
import matplotlib.image as mpimg


#from tensorflow.examples.tutorials.mnist import input_data

def readimage(folder, index):
	path = os.path.join(folder, str(index) + '.png')
	image = imageio.imread(path)
	return image

def readmask(folder, index):
	path = os.path.join(folder, str(index) + '.png')
	image = imageio.imread(path)
	return image

def readnormal(folder, index):
	path = os.path.join(folder, str(index) + '.png')
	image = imageio.imread(path)
	image = image[:,:,:]   # extract the first layer pixel of image
	return image

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	#The generated values follow a normal distribution with specified mean and standard deviation

	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)

	return tf.Variable(initial)

def conv2d(x, W, strides = [1,1,1,1]):
	# stride [1, x_movement, y_movement, 1]
	# Must have strides[0] = strides[3] = 1

	return tf.nn.conv2d(x, W, strides, padding='SAME')   # will not shrink the size



def pool(layer):
        return tf.nn.max_pool(value = layer, ksize = [1,2,2,1], strides =
                [1,2,2,1], padding = "VALID")

def upsample(layer):
        dim = layer.get_shape().as_list()
        return tf.image.resize_images(images = layer, size = [2*dim[1], 2*dim[2]])

def batch_normalize(layer):
        dim = layer.get_shape().as_list()
        batch_mean, batch_var = tf.nn.moments(layer,[0])
        scale = tf.Variable(tf.ones([dim[1], dim[2], dim[-1]]))
        beta = tf.Variable(tf.zeros([dim[1], dim[2], dim[-1]]))
        epsilon = 1e-3
        return tf.nn.batch_normalization(layer,batch_mean,batch_var,beta,scale,epsilon)





global train_num
global choose_num
train_num = 18000
choose_num = 20


image_all = np.zeros((choose_num , 128, 128, 3),dtype = float)
mask_all = np.zeros((choose_num , 128, 128, 3),dtype = float)
normal_all = np.zeros((choose_num , 128, 128, 3),dtype = float)


# define placeholder for inputs to network
color_image = tf.placeholder(tf.float32, [None, 128, 128, 3])    # 128x128x3
mask_image = tf.placeholder(tf.float32, [None, 128, 128, 3])     # 128x128x3
normal_image = tf.placeholder(tf.float32, [None, 128, 128, 3])   # 128x128x3


keep_prob = tf.placeholder(tf.float32)


###############################layer one########################################

batch_mean, batch_var = tf.nn.moments(color_image,[0])
scale = tf.Variable(tf.ones([128,128,3]))
beta = tf.Variable(tf.zeros([128,128,3]))
epsilon = 1e-3
BN = tf.nn.batch_normalization(color_image,batch_mean,batch_var,beta,scale,epsilon)

#color_image_ = tf.concat([BN, tf.expand_dims(mask_image[:,:,:,0], -1)], -1)

## minor branch 1##
W_conv1_m = weight_variable([1,1,3,128]) # patch 3x3, out size 64
b_conv1_m = bias_variable([128])
h_conv1_m = tf.nn.relu(conv2d(BN, W_conv1_m) + b_conv1_m)
print("mi1_a", h_conv1_m.get_shape())
W_conv1_n = weight_variable([3,3,128,128])
b_conv1_n = bias_variable([128])
h_conv1_n = tf.nn.relu(conv2d(h_conv1_m, W_conv1_n) + b_conv1_n)
print("mi1_b", h_conv1_n.get_shape())
W_conv1_o = weight_variable([1,1,128,256])
b_conv1_o = bias_variable([256])
h_conv1_o = tf.nn.relu(conv2d(h_conv1_n, W_conv1_o) + b_conv1_o)
print("mi1_c", h_conv1_o.get_shape())
## pooling 1##
h_conv1_p = pool(BN)
print("p1:" ,h_conv1_p.get_shape())


## main branch 2 ##
W_conv2_a = weight_variable([1,1,3,128]) # patch 3x3, out size 64
b_conv2_a = bias_variable([128])
h_conv2_a = tf.nn.relu(conv2d(h_conv1_p, W_conv2_a) + b_conv2_a)
print("ma2_a:" ,h_conv2_a.get_shape())
W_conv2_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv2_b = bias_variable([128])
h_conv2_b = tf.nn.relu(conv2d(h_conv2_a, W_conv2_b) + b_conv2_b)
print("ma2_b:" ,h_conv2_b.get_shape())
W_conv2_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv2_c = bias_variable([256])
h_conv2_c = tf.nn.relu(conv2d(h_conv2_b, W_conv2_c) + b_conv2_c)
print("ma2_c:" ,h_conv2_c.get_shape())

## minor branch 2 ##
W_conv2_m = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv2_m = bias_variable([128])
h_conv2_m = tf.nn.relu(conv2d(h_conv2_c, W_conv2_m) + b_conv2_m)
print("mi2_a:" ,h_conv2_m.get_shape())
W_conv2_n = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv2_n = bias_variable([128])
h_conv2_n = tf.nn.relu(conv2d(h_conv2_m, W_conv2_n) + b_conv2_n)
print("mi2_b:" ,h_conv2_n.get_shape())
W_conv2_o = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv2_o = bias_variable([256])
h_conv2_o = tf.nn.relu(conv2d(h_conv2_n, W_conv2_o) + b_conv2_o)
print("mi2_c:" ,h_conv2_o.get_shape())

## pooling 2##
h_conv2_p = pool(h_conv2_c)
print("p2:" ,h_conv2_p.get_shape())

## main branch 3##
W_conv3_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv3_a = bias_variable([128])
h_conv3_a = tf.nn.relu(conv2d(h_conv2_p, W_conv3_a) + b_conv3_a)
print("ma3_a:" ,h_conv3_a.get_shape())
W_conv3_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv3_b = bias_variable([128])
h_conv3_b = tf.nn.relu(conv2d(h_conv3_a, W_conv3_b) + b_conv3_b)
print("ma3_b:" ,h_conv3_b.get_shape())
W_conv3_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv3_c = bias_variable([256])
h_conv3_c = tf.nn.relu(conv2d(h_conv3_b, W_conv3_c) + b_conv3_c)
print("ma3_c:" ,h_conv3_c.get_shape())

## minor branch 3 ##
W_conv3_m = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv3_m = bias_variable([128])
h_conv3_m = tf.nn.relu(conv2d(h_conv3_c, W_conv3_m) + b_conv3_m)
print("mi3_a:" ,h_conv3_m.get_shape())
W_conv3_n = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv3_n = bias_variable([128])
h_conv3_n = tf.nn.relu(conv2d(h_conv3_m, W_conv3_n) + b_conv3_n)
print("mi3_b:" ,h_conv3_n.get_shape())
W_conv3_o = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv3_o = bias_variable([256])
h_conv3_o = tf.nn.relu(conv2d(h_conv3_n, W_conv3_o) + b_conv3_o)
print("mi3_c:" ,h_conv3_o.get_shape())

## pooling 3##
h_conv3_p = pool(h_conv3_c)
print("p3:" ,h_conv3_p.get_shape())

##main branch 4##
W_conv4_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv4_a = bias_variable([128])
h_conv4_a = tf.nn.relu(conv2d(h_conv3_p, W_conv4_a) + b_conv4_a)
print("ma3_a:" ,h_conv4_a.get_shape())
W_conv4_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv4_b = bias_variable([128])
h_conv4_b = tf.nn.relu(conv2d(h_conv4_a, W_conv4_b) + b_conv4_b)
print("ma3_b:" ,h_conv4_b.get_shape())
W_conv4_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv4_c = bias_variable([256])
h_conv4_c = tf.nn.relu(conv2d(h_conv4_b, W_conv4_c) + b_conv4_c)
print("ma3_c:" ,h_conv4_c.get_shape())
## minor branch 4 ##
W_conv4_m = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv4_m = bias_variable([128])
h_conv4_m = tf.nn.relu(conv2d(h_conv4_c, W_conv4_m) + b_conv4_m)
print("mi3_a:" ,h_conv3_m.get_shape())
W_conv4_n = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv4_n = bias_variable([128])
h_conv4_n = tf.nn.relu(conv2d(h_conv4_m, W_conv4_n) + b_conv4_n)
print("mi3_b:" ,h_conv3_n.get_shape())
W_conv4_o = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv4_o = bias_variable([256])
h_conv4_o = tf.nn.relu(conv2d(h_conv4_n, W_conv4_o) + b_conv4_o)
print("mi3_c:" ,h_conv3_o.get_shape())

## pooling 4##
h_conv4_p = pool(h_conv4_c)
print("p4:" ,h_conv4_p.get_shape())

## main branch 5##
W_conv5_a1 = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv5_a1 = bias_variable([128])
h_conv5_a1 = tf.nn.relu(conv2d(h_conv4_p, W_conv5_a1) + b_conv5_a1)
print("ma3_a1:" ,h_conv5_a1.get_shape())
W_conv5_b1 = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv5_b1 = bias_variable([128])
h_conv5_b1 = tf.nn.relu(conv2d(h_conv5_a1, W_conv5_b1) + b_conv5_b1)
print("ma3_b1:" ,h_conv5_b1.get_shape())
W_conv5_c1 = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv5_c1 = bias_variable([256])
h_conv5_c1 = tf.nn.relu(conv2d(h_conv5_b1, W_conv5_c1) + b_conv5_c1)
print("ma3_c1:" ,h_conv5_c1.get_shape())

W_conv5_a2 = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv5_a2 = bias_variable([128])
h_conv5_a2 = tf.nn.relu(conv2d(h_conv5_c1, W_conv5_a2) + b_conv5_a2)
print("ma3_a2:" ,h_conv5_a2.get_shape())
W_conv5_b2 = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv5_b2 = bias_variable([128])
h_conv5_b2 = tf.nn.relu(conv2d(h_conv5_a2, W_conv5_b2) + b_conv5_b2)
print("ma3_b2:" ,h_conv5_b2.get_shape())
W_conv5_c2 = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv5_c2 = bias_variable([256])
h_conv5_c2 = tf.nn.relu(conv2d(h_conv5_b2, W_conv5_c2) + b_conv5_c2)
print("ma3_c2:" ,h_conv5_c2.get_shape())

W_conv5_a3 = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv5_a3 = bias_variable([128])
h_conv5_a3 = tf.nn.relu(conv2d(h_conv5_c2, W_conv5_a3) + b_conv5_a3)
print("ma3_a3:" ,h_conv5_a3.get_shape())
W_conv5_b3 = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv5_b3 = bias_variable([128])
h_conv5_b3 = tf.nn.relu(conv2d(h_conv5_a3, W_conv5_b3) + b_conv5_b3)
print("ma3_b3:" ,h_conv5_b3.get_shape())
W_conv5_c3 = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv5_c3 = bias_variable([256])
h_conv5_c3 = tf.nn.relu(conv2d(h_conv5_b3, W_conv5_c3) + b_conv5_c3)
print("ma3_c3:" ,h_conv5_c3.get_shape())

## upsample 1##
h_conv5_u = upsample(h_conv5_c3)
h_conv5_u = h_conv5_u + h_conv4_o

## main branch 6##
W_conv6_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv6_a = bias_variable([128])
h_conv6_a = tf.nn.relu(conv2d(h_conv5_u, W_conv6_a) + b_conv6_a)
print("ma3_a:" ,h_conv6_a.get_shape())
W_conv6_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv6_b = bias_variable([128])
h_conv6_b = tf.nn.relu(conv2d(h_conv6_a, W_conv6_b) + b_conv6_b)
print("ma3_b:" ,h_conv6_b.get_shape())
W_conv6_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv6_c = bias_variable([256])
h_conv6_c = tf.nn.relu(conv2d(h_conv6_b, W_conv6_c) + b_conv6_c)
print("ma3_c:" ,h_conv6_c.get_shape())

## upsample 2##
h_conv6_u = upsample(h_conv6_c)
h_conv6_u = h_conv6_u + h_conv3_o

## main branch 7##
W_conv7_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv7_a = bias_variable([128])
h_conv7_a = tf.nn.relu(conv2d(h_conv6_u, W_conv7_a) + b_conv7_a)
print("ma7_a:" ,h_conv7_a.get_shape())
W_conv7_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv7_b = bias_variable([128])
h_conv7_b = tf.nn.relu(conv2d(h_conv7_a, W_conv7_b) + b_conv7_b)
print("ma7_b:" ,h_conv7_b.get_shape())
W_conv7_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv7_c = bias_variable([256])
h_conv7_c = tf.nn.relu(conv2d(h_conv7_a, W_conv7_c) + b_conv7_c)
print("ma7_c:" ,h_conv7_c.get_shape())

## upsample 3##
h_conv7_u = upsample(h_conv7_c)
h_conv7_u = h_conv7_u + h_conv2_o

## main branch 8##
W_conv8_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv8_a = bias_variable([128])
h_conv8_a = tf.nn.relu(conv2d(h_conv7_u, W_conv8_a) + b_conv8_a)
print("ma8_a:" ,h_conv8_a.get_shape())
W_conv8_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv8_b = bias_variable([128])
h_conv8_b = tf.nn.relu(conv2d(h_conv8_a, W_conv8_b) + b_conv8_b)
print("ma8_b:" ,h_conv8_b.get_shape())
W_conv8_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv8_c = bias_variable([256])
h_conv8_c = tf.nn.relu(conv2d(h_conv8_a, W_conv8_c) + b_conv8_c)
print("ma8_c:" ,h_conv8_c.get_shape())

## upsample 4##
h_conv8_u = upsample(h_conv8_c)
h_conv8_u = h_conv8_u + h_conv1_o

## concat##
color_image_ = tf.concat([h_conv8_u,BN], -1)



## minor branch 1##
W_conv10_m = weight_variable([1,1,259,128]) # patch 3x3, out size 64
b_conv10_m = bias_variable([128])
h_conv10_m = tf.nn.relu(conv2d(color_image_, W_conv10_m) + b_conv10_m)
print("mi1_a", h_conv1_m.get_shape())
W_conv10_n = weight_variable([3,3,128,128])
b_conv10_n = bias_variable([128])
h_conv10_n = tf.nn.relu(conv2d(h_conv10_m, W_conv10_n) + b_conv10_n)
print("mi1_b", h_conv1_n.get_shape())
W_conv10_o = weight_variable([1,1,128,256])
b_conv10_o = bias_variable([256])
h_conv10_o = tf.nn.relu(conv2d(h_conv10_n, W_conv10_o) + b_conv10_o)
print("mi1_c", h_conv10_o.get_shape())
## pooling 1##
h_conv10_p = pool(color_image_)
print("p1:" ,h_conv10_p.get_shape())


## main branch 2 ##
W_conv20_a = weight_variable([1,1,259,128]) # patch 3x3, out size 64
b_conv20_a = bias_variable([128])
h_conv20_a = tf.nn.relu(conv2d(h_conv10_p, W_conv20_a) + b_conv20_a)
print("ma2_a:" ,h_conv2_a.get_shape())
W_conv20_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv20_b = bias_variable([128])
h_conv20_b = tf.nn.relu(conv2d(h_conv20_a, W_conv20_b) + b_conv20_b)
print("ma2_b:" ,h_conv2_b.get_shape())
W_conv20_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv20_c = bias_variable([256])
h_conv20_c = tf.nn.relu(conv2d(h_conv20_b, W_conv20_c) + b_conv20_c)
print("ma2_c:" ,h_conv2_c.get_shape())

## minor branch 2 ##
W_conv20_m = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv20_m = bias_variable([128])
h_conv20_m = tf.nn.relu(conv2d(h_conv20_c, W_conv20_m) + b_conv20_m)
print("mi2_a:" ,h_conv20_m.get_shape())
W_conv20_n = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv20_n = bias_variable([128])
h_conv20_n = tf.nn.relu(conv2d(h_conv20_m, W_conv20_n) + b_conv20_n)
print("mi2_b:" ,h_conv2_n.get_shape())
W_conv20_o = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv20_o = bias_variable([256])
h_conv20_o = tf.nn.relu(conv2d(h_conv20_n, W_conv20_o) + b_conv20_o)
print("mi2_c:" ,h_conv2_o.get_shape())

## pooling 2##
h_conv20_p = pool(h_conv20_c)
print("p2:" ,h_conv20_p.get_shape())

## main branch 3##
W_conv30_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv30_a = bias_variable([128])
h_conv30_a = tf.nn.relu(conv2d(h_conv20_p, W_conv30_a) + b_conv30_a)
print("ma3_a:" ,h_conv3_a.get_shape())
W_conv30_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv30_b = bias_variable([128])
h_conv30_b = tf.nn.relu(conv2d(h_conv30_a, W_conv30_b) + b_conv30_b)
print("ma3_b:" ,h_conv3_b.get_shape())
W_conv30_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv30_c = bias_variable([256])
h_conv30_c = tf.nn.relu(conv2d(h_conv30_b, W_conv30_c) + b_conv30_c)
print("ma3_c:" ,h_conv3_c.get_shape())

## minor branch 3 ##
W_conv30_m = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv30_m = bias_variable([128])
h_conv30_m = tf.nn.relu(conv2d(h_conv30_c, W_conv30_m) + b_conv30_m)
print("mi3_a:" ,h_conv3_m.get_shape())
W_conv30_n = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv30_n = bias_variable([128])
h_conv30_n = tf.nn.relu(conv2d(h_conv30_m, W_conv30_n) + b_conv30_n)
print("mi3_b:" ,h_conv3_n.get_shape())
W_conv30_o = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv30_o = bias_variable([256])
h_conv30_o = tf.nn.relu(conv2d(h_conv30_n, W_conv30_o) + b_conv30_o)
print("mi3_c:" ,h_conv3_o.get_shape())

## pooling 3##
h_conv30_p = pool(h_conv30_c)
print("p3:" ,h_conv3_p.get_shape())

##main branch 4##
W_conv40_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv40_a = bias_variable([128])
h_conv40_a = tf.nn.relu(conv2d(h_conv30_p, W_conv40_a) + b_conv40_a)
print("ma3_a:" ,h_conv4_a.get_shape())
W_conv40_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv40_b = bias_variable([128])
h_conv40_b = tf.nn.relu(conv2d(h_conv40_a, W_conv40_b) + b_conv40_b)
print("ma3_b:" ,h_conv4_b.get_shape())
W_conv40_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv40_c = bias_variable([256])
h_conv40_c = tf.nn.relu(conv2d(h_conv40_b, W_conv40_c) + b_conv40_c)
print("ma3_c:" ,h_conv4_c.get_shape())
## minor branch 4 ##
W_conv40_m = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv40_m = bias_variable([128])
h_conv40_m = tf.nn.relu(conv2d(h_conv40_c, W_conv40_m) + b_conv40_m)
print("mi3_a:" ,h_conv3_m.get_shape())
W_conv40_n = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv40_n = bias_variable([128])
h_conv40_n = tf.nn.relu(conv2d(h_conv40_m, W_conv40_n) + b_conv40_n)
print("mi3_b:" ,h_conv3_n.get_shape())
W_conv40_o = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv40_o = bias_variable([256])
h_conv40_o = tf.nn.relu(conv2d(h_conv40_n, W_conv40_o) + b_conv40_o)
print("mi3_c:" ,h_conv3_o.get_shape())

## pooling 4##
h_conv40_p = pool(h_conv40_c)
print("p4:" ,h_conv4_p.get_shape())

## main branch 5##
W_conv50_a1 = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv50_a1 = bias_variable([128])
h_conv50_a1 = tf.nn.relu(conv2d(h_conv40_p, W_conv50_a1) + b_conv50_a1)
print("ma3_a1:" ,h_conv5_a1.get_shape())
W_conv50_b1 = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv50_b1 = bias_variable([128])
h_conv50_b1 = tf.nn.relu(conv2d(h_conv50_a1, W_conv50_b1) + b_conv50_b1)
print("ma3_b1:" ,h_conv5_b1.get_shape())
W_conv50_c1 = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv50_c1 = bias_variable([256])
h_conv50_c1 = tf.nn.relu(conv2d(h_conv50_b1, W_conv50_c1) + b_conv50_c1)
print("ma3_c1:" ,h_conv5_c1.get_shape())

W_conv50_a2 = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv50_a2 = bias_variable([128])
h_conv50_a2 = tf.nn.relu(conv2d(h_conv50_c1, W_conv50_a2) + b_conv50_a2)
print("ma3_a2:" ,h_conv5_a2.get_shape())
W_conv50_b2 = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv50_b2 = bias_variable([128])
h_conv50_b2 = tf.nn.relu(conv2d(h_conv50_a2, W_conv50_b2) + b_conv50_b2)
print("ma3_b2:" ,h_conv5_b2.get_shape())
W_conv50_c2 = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv50_c2 = bias_variable([256])
h_conv50_c2 = tf.nn.relu(conv2d(h_conv50_b2, W_conv50_c2) + b_conv50_c2)
print("ma3_c2:" ,h_conv5_c2.get_shape())

W_conv50_a3 = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv50_a3 = bias_variable([128])
h_conv50_a3 = tf.nn.relu(conv2d(h_conv50_c2, W_conv50_a3) + b_conv50_a3)
print("ma3_a3:" ,h_conv5_a3.get_shape())
W_conv50_b3 = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv50_b3 = bias_variable([128])
h_conv50_b3 = tf.nn.relu(conv2d(h_conv50_a3, W_conv50_b3) + b_conv50_b3)
print("ma3_b3:" ,h_conv5_b3.get_shape())
W_conv50_c3 = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv50_c3 = bias_variable([256])
h_conv50_c3 = tf.nn.relu(conv2d(h_conv50_b3, W_conv50_c3) + b_conv50_c3)
print("ma3_c3:" ,h_conv5_c3.get_shape())

## upsample 1##
h_conv50_u = upsample(h_conv50_c3)
h_conv50_u = h_conv50_u + h_conv40_o

## main branch 6##
W_conv60_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv60_a = bias_variable([128])
h_conv60_a = tf.nn.relu(conv2d(h_conv50_u, W_conv60_a) + b_conv60_a)
print("ma3_a:" ,h_conv6_a.get_shape())
W_conv60_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv60_b = bias_variable([128])
h_conv60_b = tf.nn.relu(conv2d(h_conv60_a, W_conv60_b) + b_conv60_b)
print("ma3_b:" ,h_conv6_b.get_shape())
W_conv60_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv60_c = bias_variable([256])
h_conv60_c = tf.nn.relu(conv2d(h_conv60_b, W_conv60_c) + b_conv60_c)
print("ma3_c:" ,h_conv6_c.get_shape())

## upsample 2##
h_conv60_u = upsample(h_conv60_c)
h_conv60_u = h_conv60_u + h_conv30_o

## main branch 7##
W_conv70_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv70_a = bias_variable([128])
h_conv70_a = tf.nn.relu(conv2d(h_conv60_u, W_conv70_a) + b_conv70_a)
print("ma7_a:" ,h_conv7_a.get_shape())
W_conv70_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv70_b = bias_variable([128])
h_conv70_b = tf.nn.relu(conv2d(h_conv70_a, W_conv70_b) + b_conv70_b)
print("ma7_b:" ,h_conv7_b.get_shape())
W_conv70_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv70_c = bias_variable([256])
h_conv70_c = tf.nn.relu(conv2d(h_conv70_a, W_conv70_c) + b_conv70_c)
print("ma7_c:" ,h_conv7_c.get_shape())

## upsample 3##
h_conv70_u = upsample(h_conv70_c)
h_conv70_u = h_conv70_u + h_conv20_o

## main branch 8##
W_conv80_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv80_a = bias_variable([128])
h_conv80_a = tf.nn.relu(conv2d(h_conv70_u, W_conv80_a) + b_conv80_a)
print("ma8_a:" ,h_conv8_a.get_shape())
W_conv80_b = weight_variable([3,3,128,128]) # patch 3x3, out size 64
b_conv80_b = bias_variable([128])
h_conv80_b = tf.nn.relu(conv2d(h_conv80_a, W_conv80_b) + b_conv80_b)
print("ma8_b:" ,h_conv8_b.get_shape())
W_conv80_c = weight_variable([1,1,128,256]) # patch 3x3, out size 64
b_conv80_c = bias_variable([256])
h_conv80_c = tf.nn.relu(conv2d(h_conv80_a, W_conv80_c) + b_conv80_c)
print("ma8_c:" ,h_conv8_c.get_shape())

## upsample 4##
h_conv80_u = upsample(h_conv80_c)
h_conv80_u = h_conv80_u + h_conv10_o
## main branch 9##
W_conv9_a = weight_variable([1,1,256,128]) # patch 3x3, out size 64
b_conv9_a = bias_variable([128])
h_conv9_a = tf.nn.relu(conv2d(h_conv80_u, W_conv9_a) + b_conv9_a)
print("ma8_a:" ,h_conv8_a.get_shape())
W_conv9_b = weight_variable([1,1,128,3]) # patch 3x3, out size 64
b_conv9_b = bias_variable([3])
h_conv9_b = tf.nn.relu(conv2d(h_conv9_a, W_conv9_b) + b_conv9_b)
print("ma8_b:" ,h_conv8_b.get_shape())

prediction = h_conv9_b
print("prediction:" ,prediction.get_shape())


#################################end of model##############################33

cross_entropy = tf.losses.mean_squared_error(((normal_image / 255.0) - 0.5) * 2 * mask_image / 255 \
, ((prediction / 255.0) - 0.5) * 2 * mask_image / 255)
#print(cross_entropy)
lr = 0.0005
train_step = tf.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy)
mini_loss = 10

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
save_path = "model/save_net.ckpt"
#saver.restore(sess, save_path)

prediction_folder = './train/prediction_selected'
mask_folder = './train/mask_selected'
normal_folder = './train/normal_selected'
image_folder = './train/color_selected'

image_test = np.zeros((20, 128,128,3), dtype = float)
mask_test = np.zeros((20, 128,128,3), dtype = float)
normal_test = np.zeros((20,128,128,3), dtype = float)

epochs = 100
for epoch in range(epochs):
	print("epoch: ", epoch,"############################################")
	for i in range(int(train_num/choose_num)):

		batch_index = range(i * choose_num, (i+1) * choose_num)
		print ("input images from", choose_num * i, " to ", choose_num*(i+1))
		folder_image = './train/color'
		folder_mask = './train/mask'
		folder_normal = './train/normal'
		counter = 0
		for index in batch_index:
    			image1 = readimage(folder_image, index)
	    		image_all[counter, :, :, :] = image1

    			image2 = readmask(folder_mask, index)

  	  		mask_all[counter, :, :, 0] = image2
    			mask_all[counter, :, :, 1] = image2
    			mask_all[counter, :, :, 2] = image2

   	 		image3 = readnormal(folder_normal, index)
    			normal_all[counter, :, :, :] = image3

    			counter += 1

		batch_xs = image_all[:, :, :, :]
		mask_xs = mask_all[:, :, :, :]
		batch_ys = normal_all[:, :, :, :]

		_, cross, pred = sess.run([train_step, cross_entropy, prediction],
                    feed_dict={color_image: batch_xs, mask_image: mask_xs, normal_image: batch_ys, keep_prob: 0.5})

		print("This is " + str(i) + "th iteration!!!")
		print("cross_entropy = ", cross)

	#	if i%50 == 0:
	#		print("saving model...")
	#		saver.save(sess = sess, save_path = save_path)
		# compute MAE
		##############################
#		if i%100 == 0:
#  	  		selected_dir = ['train/prediction_selected', 'train/mask_selected', 'train/normal_selected']
#
#	    		for dir in selected_dir:
# 	   			if os.path.exists(dir):
# 	   				shutil.rmtree(dir)
# 	   			os.makedirs(dir)
#
#
#	                #Save the images
#  	  		for index in range(choose_num):
#  	  			result = Image.fromarray((pred[index, :, :, :]).astype(np.uint8))
#     	                #print("result.shape ==", result.get_shape())
#  	  			result.save('train/prediction_selected/' + str(batch_index[index]) + '.png')

#   	 			mask_selected = Image.open('train/mask/' + str(batch_index[index]) + '.png')
#    				mask_selected.save('train/mask_selected/' + str(batch_index[index]) + '.png')

#  	  			mask_selected = Image.open('train/normal/' + str(batch_index[index]) + '.png')
#    				mask_selected.save('train/normal_selected/' + str(batch_index[index]) + '.png')

#  	  		prediction_folder = './train/prediction_selected'
#    			normal_folder = './train/normal_selected'
#    			mask_folder = './train/mask_selected'
#   	 		mae = Evaluation_script_.evaluate(prediction_folder, normal_folder, mask_folder,)
#    			print("MAE ==", mae)
		if i % 60 == 0 and i != 0:
			for counter_big in range(100):

				base = 18000 + counter_big * 20

				for counter_small in range(20):
					index = 18000 + counter_small + counter_big * 20
					image1 = readimage(image_folder, index)
					#print(image.get_shape())
					image_test[counter_small, :, :, :] = image1

					image2 = readmask(mask_folder, index)
					mask_test[counter_small, :, :, 0] = image2
					mask_test[counter_small, :, :, 1] = image2
					mask_test[counter_small, :, :, 2] = image2


				print(str(base) + " - " + str(base + 19)  + " images have been read!")

				batch_xs = image_test[:, :, :, :]

				mask_xs = mask_test[:, :, :, :]


				pred = sess.run(prediction, feed_dict={color_image: batch_xs,
            				mask_image: mask_xs})


				counter = 0

				for index in range(base, base + 20):
					result = Image.fromarray((pred[counter, :, :, :]).astype(np.uint8))
					result.save('train/prediction_selected/' + str(index) + '.png')
					counter += 1
				print(str(base) + " - " + str(base + 19)  + " images have been written!")
			mae = Evaluation_script_.evaluate(prediction_folder, normal_folder,mask_folder)
                        if mae<mini_loss:
				mini_loss = mae
				saver.save(sess = sess, save_path = save_path)
				print("saving model")
			saver.restore(sess,save_path)
			print("MAE ==", mae)
		#####################################
