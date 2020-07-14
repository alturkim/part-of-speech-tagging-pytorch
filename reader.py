import os
from collections import Counter
from functools import reduce
import torch
from torch.utils.data import Dataset
import conllu

import utils


class POSDataset(Dataset):
    def __init__(self, forward_char_seqs, backward_char_seqs, forward_markers_list, backward_markers_list, tag_seqs,
                 char_seqs_lengths, tag_seqs_lengths):
        self.forward_char_seqs = forward_char_seqs
        self.backward_char_seqs = backward_char_seqs
        self.forward_markers_list = forward_markers_list
        self.backward_markers_list = backward_markers_list
        self.tag_seqs = tag_seqs
        self.char_seqs_lengths = char_seqs_lengths
        self.tag_seqs_lengths = tag_seqs_lengths

        self.data_size = self.tag_seqs.size(0)

    def __getitem__(self, i):
        return self.forward_char_seqs[i], self.backward_char_seqs[i], self.forward_markers_list[i], \
               self.backward_markers_list[i], self.tag_seqs[i], self.char_seqs_lengths[i], self.tag_seqs_lengths[i]

    def __len__(self):
        return self.data_size


class DataReader:
    """
    Read and prepare dataset
    """

    def __init__(self, data_dir, config, maps_path=None):
        """create char and tag maps

        Args:
            data_dir: (string) directory containing the dataset
            config: (Config) training hyperparameters
            checkpoint_file: checkpoint file to retrieve maps
        """
        self.data_dir = data_dir
        self.config = config
        if maps_path is not None:
            maps = utils.load_checkpoint(maps_path)
            self.tag_map = maps['tag_map']
            self.char_map = maps['char_map']
        else:
            # loading training sequences
            token_seqs, tag_seqs = self.get_token_tag_seqs(os.path.join(data_dir, 'ar_padt-ud-train.conllu'))

            # create char and tag maps
            char_freq = Counter()
            tag_set = set()
            for token_seq, tag_seq in zip(token_seqs, tag_seqs):
                char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), token_seq)))
                tag_set.update(tag_seq)

            self.char_map = {k: v + 1 for v, k in
                             enumerate([char for char in char_freq.keys() if char_freq[char] >= config.min_char_freq])}
            self.tag_map = {k: v + 1 for v, k in enumerate(tag_set)}
            self.char_map['<PAD>'] = 0
            self.char_map['<END>'] = len(self.char_map)
            self.char_map['<UNK>'] = len(self.char_map)
            self.tag_map['<PAD>'] = 0
            self.tag_map['<START>'] = len(self.tag_map)
            self.tag_map['<END>'] = len(self.tag_map)

    def get_token_tag_seqs(self, data_file):
        token_seqs = []
        tag_seqs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for tokenlist in conllu.parse_incr(f):
                token_seq = []
                tag_seq = []
                for token in tokenlist:
                    if token['upos'] != '_':
                        token_seq.append(token['form'])
                        tag_seq.append(token['upos'])
                assert len(token_seq) == len(tag_seq)
                token_seqs.append(token_seq)
                tag_seqs.append(tag_seq)
            assert len(token_seqs) == len(tag_seqs)
        return token_seqs, tag_seqs

    def create_input_tensors(self, data_file, device):
        """
        Credit: Major parts of this method are adapted from Sagar Vinodababu,
        see https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/tree/041f75a37497bd1b712a426b7d18631251ecd749
        """

        token_seqs, tag_seqs = self.get_token_tag_seqs(os.path.join(self.data_dir, data_file))
        forward_char_seqs = list(
            map(lambda token_seq: list(reduce(lambda x, y: list(x) + [' '] + list(y), token_seq)) + [' '], token_seqs))
        backward_char_seqs = list(map(
            lambda token_seq: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), token_seq)))),
            token_seqs))

        # Encoding char seqs
        encoded_forward_char_seqs = list(map(lambda forward_char_seq: list(
            map(lambda c: self.char_map.get(c, self.char_map['<UNK>']), forward_char_seq)) + [self.char_map['<END>']],
                                             forward_char_seqs))
        encoded_backward_char_seqs = list(map(lambda backward_char_seq: list(
            map(lambda c: self.char_map.get(c, self.char_map['<UNK>']), backward_char_seq)) + [self.char_map['<END>']],
                                              backward_char_seqs))

        # Positions of spaces and <END> character
        forward_markers_list = list(
            map(lambda encoded_forward_char_seq: [idx for idx in range(len(encoded_forward_char_seq)) if
                                                  encoded_forward_char_seq[idx] == self.char_map[' ']] + [
                                                     len(encoded_forward_char_seq) - 1], encoded_forward_char_seqs))
        backward_markers_list = list(
            map(lambda encoded_backward_char_seq: list(reversed([idx for idx in range(len(encoded_backward_char_seq)) if
                                                                 encoded_backward_char_seq[idx] == self.char_map[
                                                                     ' ']])) + [len(encoded_backward_char_seq) - 1],
                encoded_backward_char_seqs))

        # Encode tag seqs
        encoded_tag_seqs = list(
            map(lambda tag_seq: list(map(lambda tag: self.tag_map[tag], tag_seq)) + [self.tag_map['<END>']], tag_seqs))

        # TODO
        # Since we're using CRF scores of size (prev_tags, cur_tags), find indices of target sequence in the unrolled scores
        # This will be row_index (i.e. prev_tag) * n_columns (i.e. tagset_size) + column_index (i.e. cur_tag)
        encoded_tag_seqs = list(map(
            lambda encoded_tag_seq: [self.tag_map['<START>'] * len(self.tag_map) + encoded_tag_seq[0]] + [
                encoded_tag_seq[i - 1] * len(self.tag_map) + encoded_tag_seq[i] for i in
                range(1, len(encoded_tag_seq))], encoded_tag_seqs))

        # Padding sequences
        max_char_seq_len = max(
            [len(encoded_forward_char_seq) for encoded_forward_char_seq in encoded_forward_char_seqs])
        max_tag_seq_len = max([len(encoded_tag_seq) for encoded_tag_seq in encoded_tag_seqs])

        padded_forward_char_seqs = []
        padded_backward_char_seqs = []
        padded_forward_markers_list = []
        padded_backward_markers_list = []
        padded_tag_seqs = []
        char_seqs_lengths = []
        tag_seqs_lengths = []

        for forward_char_seq, backward_char_seq, forward_markers, backward_markers, tag_seq in \
                zip(encoded_forward_char_seqs, encoded_backward_char_seqs, forward_markers_list, backward_markers_list,
                    encoded_tag_seqs):
            # Sanity  checks
            assert len(forward_markers) == len(backward_markers) == len(tag_seq)
            assert len(forward_char_seq) == len(backward_char_seq)

            # Pad
            padded_forward_char_seqs.append(
                forward_char_seq + [self.char_map['<PAD>']] * (max_char_seq_len - len(forward_char_seq)))
            padded_backward_char_seqs.append(
                backward_char_seq + [self.char_map['<PAD>']] * (max_char_seq_len - len(backward_char_seq)))
            padded_forward_markers_list.append(forward_markers + [0] * (max_tag_seq_len - len(tag_seq)))
            padded_backward_markers_list.append(backward_markers + [0] * (max_tag_seq_len - len(tag_seq)))
            padded_tag_seqs.append(tag_seq + [self.tag_map['<PAD>']] * (max_tag_seq_len - len(tag_seq)))

            char_seqs_lengths.append(len(forward_char_seq))
            tag_seqs_lengths.append(len(tag_seq))

            # Sanity check
            assert len(padded_tag_seqs[-1]) == len(padded_forward_markers_list[-1]) == len(
                padded_backward_markers_list[-1]) == max_tag_seq_len

            assert len(padded_forward_char_seqs[-1]) == len(padded_backward_char_seqs[-1]) == max_char_seq_len

        return torch.tensor(padded_forward_char_seqs, dtype=torch.int64, device=device), \
               torch.tensor(padded_backward_char_seqs, dtype=torch.int64, device=device), \
               torch.tensor(padded_forward_markers_list, dtype=torch.int64, device=device), \
               torch.tensor(padded_backward_markers_list, dtype=torch.int64, device=device), \
               torch.tensor(padded_tag_seqs, dtype=torch.int64, device=device), \
               torch.tensor(char_seqs_lengths, dtype=torch.int64, device=device), \
               torch.tensor(tag_seqs_lengths, dtype=torch.int64, device=device)
