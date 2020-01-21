# coding=utf-8

"""
Implementation of a SMILES dataset.
"""

import torch
import torch.utils.data as tud
import torch.nn.utils.rnn as tnnur


class Dataset(tud.Dataset):
    """Dataset that takes a list of SMILES only."""

    def __init__(self, smiles_list, vocabulary, tokenizer):
        """
        Instantiates a Dataset.
        :param smiles_list: A list with SMILES strings.
        :param vocabulary: A Vocabulary object.
        :param tokenizer: A Tokenizer object.
        :return:
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_list = []
        for smi in smiles_list:
            enc = self._vocabulary.encode(self._tokenizer.tokenize(smi))
            if enc is not None:
                self._encoded_list.append(enc)

    def __getitem__(self, i):
        return torch.tensor(self._encoded_list[i], dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_seqs):
        return pad_batch(encoded_seqs)


def pad_batch(encoded_seqs):
    """
    Pads a batch.
    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded.
    """
    seq_lengths = torch.tensor([len(seq) for seq in encoded_seqs], dtype=torch.int64)  # pylint: disable=not-callable
    return (tnnur.pad_sequence(encoded_seqs, batch_first=True).cuda(), seq_lengths)
