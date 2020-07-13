import argparse
import os
from utils import Config

# This script was used in an initial version of the project that uses word-level LSTM.


def extract_vocab_tags(data_dir):
    config = Config('config.json')
    tokens = []
    tags = []

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        for token_tag_pair in line.split():
            token, tag = token_tag_pair.rsplit('/', 1)
            tokens.append(token)
            tags.append(tag)
    vocab = list(set(tokens))
    vocab.insert(0, config.pad_word)
    vocab.insert(1, config.unk_word)
    tag_set = list(set(tags))

    config.dict['vocab_size'] = len(vocab)
    config.dict['tag_set_size'] = len(tag_set)
    config.save()
    print('Data statistics')
    print('Vocab size:{}\nTag set size:{}'.format(len(vocab), len(tag_set)))
    return vocab, tag_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset')
    args = parser.parse_args()
    data_dir = args.data_dir

    print('Extracting vocab and tag set...')
    vocab, tags = extract_vocab_tags(args.data_dir)
    print('Done.')

    output = ''
    for form in vocab:
        output += form + '\n'
    with open(os.path.join(data_dir, 'vocab.txt'), 'w') as f:
        f.write(output.strip())

    tag_output = ''
    for tag in tags:
        tag_output += tag + '\n'
    with open(os.path.join(data_dir, 'tag_set.txt'), 'w') as f:
        f.write(tag_output.strip())
