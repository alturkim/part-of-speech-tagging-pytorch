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
        x = x.data.contiguous()
        x = x.view(-1, self.config.lstm_hidden_dim)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def loss_fn(outputs, labels):
    """Computes the cross entropy loss, excluding loss terms for <PAD>

    Args:
        outputs: (Tensor) models outputs. Dimension: batch_size*seq_len x tag_set_size
        labels: (Tensor) ground truth label. Dimension: batch_size x seq_len. 0 for <PAD>

    Returns:
        loss: (Tensor) cross entropy loss for all tokens in the batch
    """
    labels = labels.view(-1)
    criterion = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
    loss = criterion(outputs, labels)
    return loss


def accuracy(outputs, labels):
    """Computes the accuracy given models outputs and ground truth labels, excluding terms for <PAD>

    Args:
        outputs: (Tensor) models outputs. Dimension: batch_size*seq_len x tag_set_size
        labels: (Tensor) ground truth label. Dimension: batch_size x seq_len. 0 for <PAD>

    Returns:
        accuracy: (float)
    """
    labels = labels.view(-1)
    mask = (labels > 0).float()
    outputs = torch.argmax(outputs, dim=1)
    return torch.sum(outputs == labels)/float(torch.sum(mask))
