import os
import argparse
import logging

from tqdm import trange
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import utils
from utils import Config
from reader import DataReader, POSDataset
from models.net import Net
from models import net


def parse_arguments(parser):
    parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
    parser.add_argument('--config_dir', default='config/', help='Directory containing config.json')
    parser.add_argument('--restore_file', help='Optional, name of the file in saved_models directory containing \
                        pre-trained model to be loaded before training')

    args = parser.parse_args()
    for k, v in vars(args).items():
        logging.info(k + ' : ' + str(v))
    return args


def train(model, optimizer, loss_fn, data_loader, accuracy_fn, config, epochs):
    """Trains the model `batches` number of time

    Args:
        model: (torch.nn.Module) an instance of the model class.
        optimizer: (torch.optim) optimizer for model parameters
        loss_fn:
        data_loader: (torch.util.data.DataLoader)
        metrics: (list) a list of metric to evaluate the model
        config: (Config) a config object containing the model hyperparameters
        epochs: (int) number of batches to train on.
    """

    # set the model to training mode
    model.train()

    losses = []
    avg_loss = utils.RunningAverage()
    avg_train_acc = utils.RunningAverage()

    for epoch in trange(epochs):
        avg_loss.reset()
        avg_train_acc.reset()
        for seqs, labels, lengths in data_loader:
            # compute model output and loss
            outputs = model(seqs, lengths)
            labels = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False).data
            loss = loss_fn(outputs, labels)
            train_acc = accuracy_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss))
            avg_loss.update(float(loss))
            avg_train_acc.update(float(train_acc))
        logging.info('Epoch:' + str(epoch))
        logging.info('Loss:' + str(avg_loss.avg))
        logging.info('Training Accuracy' + str(avg_train_acc.avg))


if __name__ == '__main__':
    args = parse_arguments(argparse.ArgumentParser())
    data_dir = args.data_dir
    config_dir = args.config_dir
    utils.set_logger(os.path.join(config_dir, 'logging.conf'))
    config = Config(os.path.join(config_dir, 'config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reader = DataReader('./data/', config)
    dataset = POSDataset(*reader.create_input_tensors('train.txt'))
    data_loader = DataLoader(dataset, batch_size=2000)
    model = Net(config)
    metric = net.accuracy
    loss_fn = net.loss_fn
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate)

    train(model, optimizer, loss_fn, data_loader, metric, config, 1000)


