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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Training on GPU ... ')


def parse_arguments(parser):
    parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
    parser.add_argument('--config_dir', default='config/', help='Directory containing config.json')
    parser.add_argument('--checkpoint_dir', default='checkpoints/', help='Directory to save and load models parameters.')

    args = parser.parse_args()
    for k, v in vars(args).items():
        logging.info(k + ' : ' + str(v))
    return args


def train(model, optimizer, criterion, data_loader):
    """Trains the model for one epoch

    Args:
        model: (torch.nn.Module) an instance of the model class.
        optimizer: (torch.optim) optimizer for model parameters
        criterion: a loss function
        data_loader: (torch.util.data.DataLoader)
    """

    # set the model to training mode
    model.train()

    avg_loss = utils.RunningAverage()
    avg_acc = utils.RunningAverage()

    for seqs, labels, lengths in data_loader:
        outputs = model(seqs, lengths)
        labels = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False).data
        loss = criterion(outputs, labels.view(-1))
        acc = net.accuracy(outputs, labels)
        avg_loss.update(float(loss))
        avg_acc.update(float(acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return avg_loss.avg, avg_acc.avg


def evaluate(model, criterion, data_loader):
    """Evaluates the model over all data in data_loader

    Args:
        model: (torch.nn.Module) an instance of the model class.
        criterion: a loss function
        data_loader: (torch.util.data.DataLoader)
    """
    # set the model to evaluation mode
    model.eval()

    for seqs, labels, lengths in data_loader:
        # compute model output and loss
        outputs = model(seqs, lengths)
        labels = pack_padded_sequence(labels, lengths, batch_first=True, enforce_sorted=False).data
        loss = criterion(outputs, labels.view(-1))
        acc = net.accuracy(outputs, labels)
    return float(loss), float(acc)


def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, config, checkpoint_dir, checkpoint_file=None):
    """Trains and evaluates the model for `epochs` times

    Args:
        model: (torch.nn.Module) an instance of the model class.
        optimizer: (torch.optim) optimizer for model parameters
        criterion: a loss function
        train_loader: (torch.util.data.DataLoader) loader for training data
        val_loader: (torch.util.data.DataLoader) loader for validation data
        config: (Config) a config object containing the model hyperparameters
        checkpoint_dir: (str) directory to save checkpoint files in
        checkpoint_file: (str) file name of checkpoint to be loaded before training
    """

    if checkpoint_file is not None:
        utils.load_checkpoint(checkpoint_file, model, optimizer)

    losses = []
    best_val_acc = 0

    for epoch in trange(config.epochs):
        loss, train_acc = train(model, optimizer, criterion, train_loader)
        losses.append(float(loss))
        _, val_acc = evaluate(model, criterion, val_loader)

        logging.info('Epoch: ' + str(epoch) + ' Loss: ' + str(loss) \
                     + '\nTraining Accuracy: ' + str(train_acc) + ' Validation Accuracy: ' + str(val_acc))

        is_best = val_acc > best_val_acc
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()}, is_best, checkpoint_dir)


if __name__ == '__main__':
    args = parse_arguments(argparse.ArgumentParser())
    data_dir = args.data_dir
    config_dir = args.config_dir
    checkpoint_dir = args.checkpoint_dir
    utils.set_logger(os.path.join(config_dir, 'logging.conf'))
    config = Config(os.path.join(config_dir, 'config.json'))

    reader = DataReader(data_dir, config)
    train_dataset = POSDataset(*reader.create_input_tensors('train.txt', device))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_dataset = POSDataset(*reader.create_input_tensors('dev.txt', device))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = Net(config).to(device)
    criterion = torch.nn.NLLLoss(ignore_index=0, reduction='mean').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, config, checkpoint_dir)
    # os.path.join(checkpoint_dir, 'checkpoint.pth.tar')




