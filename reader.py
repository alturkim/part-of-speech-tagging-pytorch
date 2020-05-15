import os
import torch
from torch.utils.data import Dataset

from utils import Config


class POSDataset(Dataset):
    def __init__(self, sentences, labels, lengths):
        self.sentences = sentences
        self.labels = labels
        self.lengths = lengths
        self.data_size = self.sentences.size(0)

    def __getitem__(self, i):
        return self.sentences[i], self.labels[i], self.lengths[i]

    def __len__(self):
        return self.data_size


class DataReader:
    """
    Read and prepare dataset
    """
    def __init__(self, data_dir, config):
        """Loads vocab and tag set

        Args:
            data_dir: (string) directory containing the dataset, vocab, and tag set
            config: (Config) training hyperparameters
        """
        self.data_dir = data_dir
        # loading data config
        self.config = Config('config/config.json')

        # loading vocab and tag set
        with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
            content = f.readlines()
        self.vocab = {k.strip(): v for v, k in enumerate(content)}

        with open(os.path.join(data_dir, 'tag_set.txt'), 'r') as f:
            content = f.readlines()
        self.tag_set = {k.strip(): v for v, k in enumerate(content)}

        self.unk_idx = self.vocab[self.config.unk_word]
        self.padword_idx = self.vocab[self.config.pad_word]

    def create_input_tensors(self, data_file):
        sentences = []
        labels = []
        with open(os.path.join(self.data_dir, data_file), 'r') as f:
            lines = f.readlines()
        for line in lines:
            sentence = []
            label = []
            for token_tag_pair in line.split():
                token, tag = token_tag_pair.rsplit('/', 1)
                sentence.append(self.vocab.get(token, self.unk_idx))
                label.append(self.tag_set[tag])
            # sanity check
            assert len(sentence) == len(label)
            sentences.append(sentence)
            labels.append(label)
        assert len(sentences) == len(labels)

        max_len = max([len(sentence) for sentence in sentences])
        assert max_len == max([len(label) for label in labels])

        padded_sentences = []
        padded_labels = []
        sentences_lengths = []

        for sentence, label in zip(sentences, labels):
            padded_sentences.append(sentence + [self.padword_idx] * (max_len - len(sentence)))
            padded_labels.append(label + [self.padword_idx] * (max_len - len(label)))
            sentences_lengths.append(len(sentence))

        return torch.LongTensor(padded_sentences), torch.LongTensor(padded_labels), torch.LongTensor(sentences_lengths)
