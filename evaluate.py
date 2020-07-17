import argparse
import os

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models.net import ViterbiLoss
import utils
from utils import Config
from reader import POSDataset, DataReader
from models.viterbi_decoder import  ViterbiDecoder


def parse_arguments(parser):
    parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
    parser.add_argument('--config_dir', default='config/', help='Directory containing config.json')
    parser.add_argument('--checkpoint_dir', default='checkpoints/',
                        help='Directory to load models parameters from.')
    parser.add_argument('--checkpoint_file', default='BEST_checkpoint.pth.tar', \
                        help='Checkpoint file containing models parameters.')
    parser.add_argument('--maps_file', default='maps.pth.tar', help='Checkpoint file containing maps of training data.')

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(k + ' : ' + str(v))
    return args


def evaluate(model, criterion, data_loader, viterbi_decoder):
    """Evaluates the model over all data in data_loader

    Args:
        model: (torch.nn.Module) an instance of the model class.
        criterion: a loss function
        data_loader: (torch.util.data.DataLoader)
    """
    # set the model to evaluation mode
    model.eval()

    avg_loss = utils.RunningAverage()
    avg_score = utils.RunningAverage()

    for forward_char_seqs, backward_char_seqs, forward_markers_list, backward_markers_list, tag_seqs, char_seqs_lengths, tag_seqs_lengths in data_loader:
        crf_scores, tag_seqs, tag_seqs_lengths = model(forward_char_seqs, backward_char_seqs, forward_markers_list,
                                                       backward_markers_list, tag_seqs, char_seqs_lengths,
                                                       tag_seqs_lengths)
        loss = criterion(crf_scores, tag_seqs, tag_seqs_lengths)
        decoded = viterbi_decoder.decode(crf_scores.to("cpu"), tag_seqs_lengths.to("cpu"))

        lengths = tag_seqs_lengths - 1
        lengths = lengths.tolist()
        decoded = pack_padded_sequence(decoded, lengths, batch_first=True).data
        tag_seqs = tag_seqs % viterbi_decoder.tag_set_size
        tag_seqs = pack_padded_sequence(tag_seqs, lengths, batch_first=True).data

        f1 = f1_score(tag_seqs.to("cpu").numpy(), decoded.numpy(), average='macro')

        avg_loss.update(loss.item(), crf_scores.size(0))
        avg_score.update(f1, sum(lengths))
    return float(avg_loss.avg), float(avg_score.avg)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Evaluating on GPU ... ')
    else:
        device = torch.device('cpu')
        print('Evaluating on CPU ... ')

    args = parse_arguments(argparse.ArgumentParser())
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    checkpoint_file = args.checkpoint_file
    maps_file = args.maps_file
    config_dir = args.config_dir
    config = Config(os.path.join(config_dir, 'config.json'))

    reader = DataReader(data_dir, config, os.path.join(checkpoint_dir, maps_file))
    dataset = POSDataset(*reader.create_input_tensors('ar_padt-ud-test.conllu', device))
    loader = DataLoader(dataset)

    checkpoint = utils.load_checkpoint(os.path.join(checkpoint_dir,checkpoint_file))
    model = checkpoint['model']
    viterbi_decoder = ViterbiDecoder(reader.tag_map)
    criterion = ViterbiLoss(reader.tag_map).to(device)
    print('Starting Evaluation ...')
    loss, f1 = evaluate(model, criterion, loader, viterbi_decoder)
    print('Evaluation done ...')
    print('F1 score on test data: ', f1)

