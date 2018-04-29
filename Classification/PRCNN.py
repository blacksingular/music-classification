"""Parelleling Bi-RNN + CNN according to paper 1712.08370

Routine:
    1. Load STFT spectrum from GTZAN: [N, 128, 513]
    2.
        Path A:
            cnn(16, 3, 3, 1, 1) - pooling - cnn(32, 3, 3, 1, 1) - 
            pooling - cnn(64...) - pooling - cnn(128...) - pooling - 
            cnn(64...) - pooling - flattened output [N, 256]
        Path B:
            pooling(1, 2) strides(1, 2) to [N, 128, 256] - 
            embedding to [N, 128, 128] - BGRU-RNN - output [N, 256]
    3. Concate output A and B [N, 512]
    4. Dense(10)
    5. Softmax(10)

"""

# pylint: disable=missing-docstring

import os
import re
import sys
import tarfile

import tensorflow as tf

import GTZAN as G

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of samples to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/GTZAN',
                           """Path to the GTZAN data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants
IMAGE_SIZE = G.IMAGE_SIZE
NUM_CLASSES = G.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = G.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = G.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'



class PRCNN():
    """Parelleling Bi-RNN + CNN"""

    def __init__(self):
        pass
