
import tensorflow as tf
from PRCNN import PRCNN


def main():
    model = PRCNN()
    print('PRCNN')
    model.Build(mode=tf.estimator.ModeKeys.TRAIN)
    print('builded')
    model.train()
    print('trained')


main()