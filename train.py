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


def train(model, optimizer, loss_fn, data_loader, accuracy_fn, config):
    """Trains the model for one epoch

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

    for seqs, labels, lengths in data_loader:
        # compute model output and loss
        outputs = model(seqs, lengths)
        labels = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False).data
        loss = loss_fn(outputs, labels)
        acc = accuracy_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss, acc


def evaluate(model, loss_fn, data_loader, accuracy_fn):
    """Evaluates the model over all data in data_loader

    Args:
        model: (torch.nn.Module) an instance of the model class.
        loss_fn:
        data_loader: (torch.util.data.DataLoader)
        accuracy_fn: (list) a list of metric to evaluate the model
    """
    # set the model to evaluation mode
    model.eval()

    for seqs, labels, lengths in data_loader:
        # compute model output and loss
        outputs = model(seqs, lengths)
        labels = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False).data
        loss = loss_fn(outputs, labels)
        acc = accuracy_fn(outputs, labels)
    return float(loss), float(acc)


def train_and_evaluate(model, optimizer, loss_fn, accuracy_fn, train_loader, val_loader, config, epochs, checkpoint_file=None):
    """Trains and evaluates the model for `epochs` times

    Args:
        model: (torch.nn.Module) an instance of the model class.
        optimizer: (torch.optim) optimizer for model parameters
        loss_fn:
        data_loader: (torch.util.data.DataLoader)
        metrics: (list) a list of metric to evaluate the model
        config: (Config) a config object containing the model hyperparameters
        epochs: (int) number of batches to train on.
        checkpoint_file: (str) file name of checkpoint to be loaded before training
    """

    if checkpoint_file is not None:
        utils.load_checkpoint(checkpoint_file, model)

    losses = []
    avg_loss = utils.RunningAverage()
    avg_train_acc = utils.RunningAverage()
    best_val_acc = 0

    for epoch in trange(epochs):
        avg_loss.reset()
        avg_train_acc.reset()
        loss, train_acc = train(model, optimizer, loss_fn, train_loader, metric, config)
        losses.append(float(loss))
        avg_loss.update(float(loss))
        avg_train_acc.update(float(train_acc))

        _, val_acc = evaluate(model, loss_fn, val_loader, metric)

        logging.info('Epoch: ' + str(epoch) + ' Loss: ' + str(avg_loss.avg) \
                     + '\nTraining Accuracy: ' + str(avg_train_acc.avg) + ' Validation Accuracy: ' + str(val_acc))

        is_best = val_acc > best_val_acc

        utils.save_checkpoint(state={'epoch': epoch + 1, 'state_dict': model.state_dict()}, is_best=is_best)


if __name__ == '__main__':
    args = parse_arguments(argparse.ArgumentParser())
    data_dir = args.data_dir
    config_dir = args.config_dir
    utils.set_logger(os.path.join(config_dir, 'logging.conf'))
    config = Config(os.path.join(config_dir, 'config.json'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reader = DataReader('./data/', config)
    train_dataset = POSDataset(*reader.create_input_tensors('train.txt'))
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_dataset = POSDataset(*reader.create_input_tensors('dev.txt'))
    val_loader = DataLoader(val_dataset, batch_size=1000)

    model = Net(config)
    metric = net.accuracy
    loss_fn = net.loss_fn
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate)

    train_and_evaluate(model, optimizer, loss_fn, metric, train_loader, val_loader, config, 1000, 'checkpoint.pth.tar')




