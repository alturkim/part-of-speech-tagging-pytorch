import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.lstm_hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.config.lstm_hidden_dim*2, self.config.tag_set_size)

    def forward(self, x, x_lengths):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = x.data
        x = self.dropout(x)
        x = x.contiguous()
        x = x.view(-1, self.config.lstm_hidden_dim*2)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CRF(nn.Module):
    def __init__(self, hidden_dim, tagset_size):
        """
        Args:
            hidden_dim: (int) dim of LSTM output
            target_size: (int) number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.target_size)
        self.transition = nn.Parameter(torch.zeros((self.target_size, self.target_size), dtype=torch.float64))

    def forward(self, x):
        """
        Args:
            x: (tensor) output of LSTM, dimension (batch_size, timesteps, hidden_dim)

        Returns:
            tensor: CRF scores, dimension (batch_size, timesteps, target_size, target_size)
        """
        batch_size = x.size(0)
        timesteps = x.size(1)
        emission_scores = self.emission(x).unsqueeze(2).expand(batch_size, timesteps, self.tagset_size, self.tagset_size)
        crf_score = emission_scores + self.transition.unsqueeze(0).unsqueeze(0)
        return crf_score



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
