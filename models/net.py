import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.lstm_hidden_dim, bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.config.lstm_hidden_dim * 2, self.config.tag_set_size)

    def forward(self, x, x_lengths):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = x.data
        x = self.dropout(x)
        x = x.contiguous()
        x = x.view(-1, self.config.lstm_hidden_dim * 2)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CRF(nn.Module):
    def __init__(self, hidden_dim, tag_set_size):
        """
        Args:
            hidden_dim: (int) dim of LSTM output
            tag_set_size: (int) number of tags
        """
        super(CRF, self).__init__()
        self.tag_set_size = tag_set_size
        self.emission = nn.Linear(hidden_dim, self.tag_set_size)
        self.transition = nn.Parameter(torch.zeros((self.tag_set_size, self.tag_set_size), dtype=torch.float64))

    def forward(self, x):
        """
        Args:
            x: (tensor) output of LSTM, dimension (batch_size, timesteps, hidden_dim)

        Returns:
            tensor: CRF scores, dimension (batch_size, timesteps, tag_set_size, tag_set_size)
        """
        batch_size = x.size(0)
        timesteps = x.size(1)
        emission_scores = self.emission(x).unsqueeze(2).expand(batch_size, timesteps, self.tag_set_size,
                                                               self.tag_set_size)
        crf_score = emission_scores + self.transition.unsqueeze(0).unsqueeze(0)
        return crf_score


class LSTM_CRF(nn.Module):
    def __init__(self, tag_set_size, char_set_size, config):
        super(LSTM_CRF, self).__init__()

        self.tag_set_size = tag_set_size
        self.char_set_size = char_set_size
        self.embedding_dim = config.embedding_dim
        self.lstm_hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.num_layers
        self.dropout = nn.Dropout(p=config.dropout)

        self.embedding = nn.Embedding(self.char_set_size, self.embedding_dim)
        self.forward_lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, num_layers=self.num_layers,
                                    dropout=config.dropout)
        self.backward_lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, num_layers=self.num_layers,
                                     dropout=config.dropout)
        self.crf = CRF(2*self.lstm_hidden_dim, self.tag_set_size)

    def forward(self, padded_forward_char_seqs, padded_backward_char_seqs, padded_forward_markers_list,
                padded_backward_markers_list, padded_tag_seqs, char_seqs_lengths, tag_seqs_lengths):
        batch_size = padded_forward_char_seqs.size(0)
        max_tag_seq_len = padded_tag_seqs.size(1)

        # Sort by length
        char_seqs_lengths, char_sort_idx = char_seqs_lengths.sort(dim=0, descending=True)
        padded_forward_char_seqs = padded_forward_char_seqs[char_sort_idx]
        padded_backward_char_seqs = padded_backward_char_seqs[char_sort_idx]
        padded_forward_markers_list = padded_forward_markers_list[char_sort_idx]
        padded_backward_markers_list = padded_backward_markers_list[char_sort_idx]
        padded_tag_seqs = padded_tag_seqs[char_sort_idx]
        tag_seqs_lengths = tag_seqs_lengths[char_sort_idx]

        forward_embeddings = self.embedding(padded_forward_char_seqs)
        backward_embeddings = self.embedding(padded_backward_char_seqs)

        forward_embeddings = self.dropout(forward_embeddings)
        backward_embeddings = self.dropout(backward_embeddings)

        forward_embeddings = pack_padded_sequence(forward_embeddings, char_seqs_lengths.tolist(), batch_first=True)
        backward_embeddings = pack_padded_sequence(backward_embeddings, char_seqs_lengths.tolist(), batch_first=True)

        forward_hidden, _ = self.forward_lstm(forward_embeddings)
        backward_hidden, _ = self.backward_lstm(backward_embeddings)

        forward_hidden, _ = pad_packed_sequence(forward_hidden, batch_first=True)
        backward_hidden, _ = pad_packed_sequence(backward_hidden, batch_first=True)

        # Sanity check
        assert forward_hidden.size(1) == max(char_seqs_lengths.tolist()) == list(char_seqs_lengths)[0]

        padded_forward_markers_list = padded_forward_markers_list.unsqueeze(2).expand(batch_size, max_tag_seq_len,
                                                                                      self.lstm_hidden_dim)
        padded_backward_markers_list = padded_backward_markers_list.unsqueeze(2).expand(batch_size,
                                                                                        max_tag_seq_len,
                                                                                        self.lstm_hidden_dim)
        forward_hidden_selected = torch.gather(forward_hidden, 1, padded_forward_markers_list)
        backward_hidden_selected = torch.gather(backward_hidden, 1, padded_backward_markers_list)

        tag_seqs_lengths, tag_seqs_sort_idxs = tag_seqs_lengths.sort(dim=0, descending=True)
        padded_tag_seqs = padded_tag_seqs[tag_seqs_sort_idxs]
        forward_hidden_selected = forward_hidden_selected[tag_seqs_sort_idxs]
        backward_hidden_selected = backward_hidden_selected[tag_seqs_sort_idxs]

        combined_hidden = torch.cat((forward_hidden_selected, backward_hidden_selected), dim=2)

        # print('lstm_hidden_dim', self.lstm_hidden_dim)
        # print('padded_forward_char_seqs', padded_forward_char_seqs.size())
        # print('padded_forward_markers_list', padded_forward_markers_list.size())
        # print('padded_tag_seqs', padded_tag_seqs.size())
        # print('forward_hidden_selected', forward_hidden_selected.size())
        # print('backward_hidden_selected', backward_hidden_selected.size())
        # print('combined_hidden', combined_hidden.size())


        crf_scores = self.crf(combined_hidden)

        return crf_scores, padded_tag_seqs, tag_seqs_lengths


class ViterbiLoss(nn.Module):
    def __init__(self, tag_map):
        super(ViterbiLoss, self).__init__()
        self.tag_set_size = len(tag_map)
        self.start_tag = tag_map['<START>']
        self.end_tag = tag_map['<END>']

    def forward(self, scores, targets, lengths):
        """
        Args:
            scores: CRF scores
            targets: true tags (idxs) in unrolled CRF scores
            lengths: seq length

        Returns:
            viterbi_loss: (float)
        """

        batch_size = scores.size(0)
        max_tag_seq_len = scores.size(1)

        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size, max_tag_seq_len, -1), 2, targets).squeeze(2)
        scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True).data
        true_scores = scores_at_targets.sum()

        aggregate_scores = torch.zeros(batch_size, self.tag_set_size, device=scores.device)
        for t in range(max(lengths)):
            effective_batch_size = sum([seq_len > t for seq_len in lengths])
            if t == 0:
                aggregate_scores[:effective_batch_size] = scores[:effective_batch_size, t, self.start_tag, :]
            else:
                aggregate_scores[:effective_batch_size] = torch.logsumexp(
                    scores[:effective_batch_size, t, :, :] + aggregate_scores[:effective_batch_size].unsqueeze(2),
                    dim=1)

        final_scores = aggregate_scores[:, self.end_tag].sum()

        viterbi_loss = final_scores - true_scores
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss


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
    return torch.sum(outputs == labels) / float(torch.sum(mask))
