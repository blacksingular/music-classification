import os

import tensorflow as tf
import argparse
from PRCNN import PRCNN


def parse_args():
    parser = argparse.ArgumentParser(description='train or test')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()
    return args


def main(args):
    model = PRCNN()
    print('PRCNN')
    model.Build()
    print('builded')
    if args.mode == 'train':
        model.train()
        print('trained')
    else:
        model.test()
        print('finished')


if __name__ == '__main__':
    args = parse_args()
    main(args)
