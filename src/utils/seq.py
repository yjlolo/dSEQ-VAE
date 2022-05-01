import numpy as np
from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence


def reverse_sequence(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    x: tensor (b, T_max, input_dim)
    seq_lengths: tensor (b, )

    Returns
    -------
    x_reverse: tensor (b, T_max, input_dim)
        The input x in reversed order w.r.t. time-axis
    """
    x_reverse = torch.zeros_like(x)
    for b in range(x.size(0)):
        t = seq_lengths[b]
        time_slice = torch.arange(t - 1, -1, -1, device=x.device)
        reverse_seq = torch.index_select(x[b], 0, time_slice)
        x_reverse[b][:t] = reverse_seq

    return x_reverse


def pad_and_reverse(rnn_output, seq_lengths, total_length):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    rnn_output: tensor  # shape to be confirmed, should be packed rnn output
    seq_lengths: tensor (b, )

    Returns
    -------
    reversed_output: tensor (b, T_max, input_dim)
        The input sequence, unpacked and padded,
        in reversed order w.r.t. time-axis
    """
    rnn_output, _ = pad_packed_sequence(
        rnn_output, batch_first=True, total_length=total_length
    )
    reversed_output = reverse_sequence(rnn_output, seq_lengths)
    return reversed_output


def get_mini_batch_mask(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    x: tensor (b, T_max, input_dim)
    seq_lengths: tensor (b, )

    Returns
    -------
    mask: tensor (b, T_max)
        A binary mask generated according to `seq_lengths`
    """
    batch, t_max = x.size(0), x.size(1)
    mask = torch.zeros([batch, t_max]).type_as(x)
    for b in range(batch):
        mask[b, :seq_lengths[b]] = 1
    return mask


def seq_collate_fn(batch):
    """
    A customized `collate_fn` for loading variable-length sequential data
    """
    idx, seq, seq_lengths, labels = zip(*batch)

    idx = torch.as_tensor(idx)
    seq_lengths = torch.as_tensor(seq_lengths)
    labels = torch.as_tensor(np.array(labels))
    # Check if all sequences share a same sequence length
    if seq_lengths.sum() != seq_lengths[0] * len(seq_lengths):
        seq = pad_sequence(seq, batch_first=True)
        _, sorted_seq_length_indices = torch.sort(seq_lengths)
        sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
        seq_lengths = seq_lengths[sorted_seq_length_indices]
        seq = seq[sorted_seq_length_indices]
        labels = labels[sorted_seq_length_indices]
        mask = get_mini_batch_mask(seq, seq_lengths)
    else:
        # We can directly stack the list of batched sequences
        # if all of them share a same sequence length
        seq = torch.stack(seq, dim=0)
        mask = torch.ones(len(seq_lengths), seq_lengths[0]).type_as(seq)

    return seq, mask, seq_lengths, labels


def pad_batch(batch_list: List[Dict], kw: str, pad_to: int):
    return torch.cat(
        [
            F.pad(
            d_i[kw], pad=(0, 0, 0, pad_to - d_i[kw].size(1))
            ) for d_i in batch_list
        ]
    )
