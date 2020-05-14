import os
import argparse
import logging

from torch.utils.data import DataLoader
import utils
from utils import Config
from reader import DataReader, POSDataset
from net import Net


def parse_arguments(parser):
    parser.add_argument('--data_dir', default='data/')

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(k + ' : ' + str(v))
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    data_dir = args.data_dir
    utils.set_logger('logging.conf')
    config = Config('config.json')
    # for k, v in config.dict.items():
    #     logging.info((k, v))

    reader = DataReader('./data/', Config('config.json'))
    dataset = POSDataset(*reader.create_input_tensors('train.txt'))
    dataloader = DataLoader(dataset, batch_size=2)
    model = Net(config)


