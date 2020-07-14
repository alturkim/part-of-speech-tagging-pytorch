import os
import argparse
import logging

from tqdm import trange
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import f1_score

import utils
from utils import Config
from reader import DataReader, POSDataset
from models.net import Net, ViterbiLoss, LSTM_CRF
from models.viterbi_decoder import ViterbiDecoder
from evaluate import evaluate


def parse_arguments(parser):
    parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
    parser.add_argument('--config_dir', default='config/', help='Directory containing config.json')
    parser.add_argument('--checkpoint_dir', default='checkpoints/',
                        help='Directory to save and load models parameters.')
    parser.add_argument('--checkpoint_file', help='Checkpoint file containing models parameters.')
    parser.add_argument('--maps_file', help='Checkpoint file containing maps of training data.')

    args = parser.parse_args()
    for k, v in vars(args).items():
        logging.info(k + ' : ' + str(v))
    return args


def train(model, optimizer, criterion, data_loader, epoch, viterbi_decoder):
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
    avg_score = utils.RunningAverage()

    for forward_char_seqs, backward_char_seqs, forward_markers_list, backward_markers_list, tag_seqs, char_seqs_lengths, tag_seqs_lengths in data_loader:
        crf_scores, tag_seqs, tag_seqs_lengths = model(forward_char_seqs, backward_char_seqs, forward_markers_list,
                                                       backward_markers_list, tag_seqs, char_seqs_lengths,
                                                       tag_seqs_lengths)
        loss = criterion(crf_scores, tag_seqs, tag_seqs_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        decoded = viterbi_decoder.decode(crf_scores.to("cpu"), tag_seqs_lengths.to("cpu"))

        lengths = tag_seqs_lengths - 1
        lengths = lengths.tolist()
        decoded = pack_padded_sequence(decoded, lengths, batch_first=True).data
        tag_seqs = tag_seqs % viterbi_decoder.tag_set_size
        tag_seqs = pack_padded_sequence(tag_seqs, lengths, batch_first=True).data

        f1 = f1_score(tag_seqs.to("cpu").numpy(), decoded.numpy(), average='macro')

        avg_loss.update(loss.item(), crf_scores.size(0))
        avg_score.update(f1, sum(lengths))

    return avg_loss.avg, avg_score.avg


def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, start_epoch, best_f1, viterbi_decoder,
                       config, checkpoint_dir, checkpoint_file=None):
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
    lr = config.lr
    losses = []

    for epoch in trange(start_epoch, config.epochs):
        loss, train_score = train(model, optimizer, criterion, train_loader, epoch, viterbi_decoder)
        losses.append(float(loss))
        _, val_score = evaluate(model, criterion, val_loader, viterbi_decoder)

        logging.info(
            'Epoch: ' + str(epoch) + '\nTraining Loss: ' + str(train_score) + ' Validation Score: ' + str(val_score))

        is_best = val_score > best_f1
        best_f1 = max(val_score, best_f1)

        utils.save_checkpoint(
            {'epoch': epoch, 'model': model, 'optimizer': optimizer, 'val_score': val_score}, is_best, checkpoint_dir)

        utils.adjust_learning_rate(optimizer, epoch, lr, config.lr_decay)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Training on GPU ... ')
    else:
        device = torch.device('cpu')
        print('Training on CPU')

    args = parse_arguments(argparse.ArgumentParser())
    data_dir = args.data_dir
    config_dir = args.config_dir
    checkpoint_dir = args.checkpoint_dir
    checkpoint_file = args.checkpoint_file
    maps_file = args.maps_file
    utils.set_logger(os.path.join(config_dir, 'logging.conf'))
    config = Config(os.path.join(config_dir, 'config.json'))

    start_epoch = 0
    best_f1 = 0

    if maps_file is not None:
        reader = DataReader(data_dir, config, os.path.join(checkpoint_dir, maps_file))
    else:
        reader = DataReader(data_dir, config)
        maps = {'char_map': reader.char_map, 'tag_map': reader.tag_map}
        utils.save_maps(maps, checkpoint_dir)

    if checkpoint_file is not None:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = utils.load_checkpoint(checkpoint_path)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']
    else:
        # model = Net(config).to(device)
        model = LSTM_CRF(tag_set_size=len(reader.tag_map), char_set_size=len(reader.char_map), config=config)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                                    momentum=config.momentum)

    criterion = ViterbiLoss(reader.tag_map).to(device)
    viterbi_decoder = ViterbiDecoder(reader.tag_map)

    train_dataset = POSDataset(*reader.create_input_tensors('ar_padt-ud-train.conllu', device))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = POSDataset(*reader.create_input_tensors('ar_padt-ud-dev.conllu', device))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, start_epoch, best_f1, viterbi_decoder,
                       config, checkpoint_dir)
    # os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
