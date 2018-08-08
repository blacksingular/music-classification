import os

import tensorflow as tf

from PRCNN import PRCNN


def main():
    model = PRCNN()
    print('PRCNN')
    model.Build()
    print('builded')
    model.train()
    print('trained')


main()
