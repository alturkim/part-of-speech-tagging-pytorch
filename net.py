import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.config.lstm_hidden_dim, self.config.tag_set_size)

    def forward(self, x, x_lengths):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x.contiguous()
        x = x.view(-1, self.config.lstm_hidden_dim)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def loss_fn():
    pass


def accuracy():
    pass