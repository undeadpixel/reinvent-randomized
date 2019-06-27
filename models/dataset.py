# coding=utf-8

"""
Implementation of a SMILES dataset.
"""

import torch
import torch.utils.data as tud


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset."""

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
        self._smiles_list = list(smiles_list)

    def __getitem__(self, i):
        smi = self._smiles_list[i]
        tokens = self._tokenizer.tokenize(smi)
        encoded = self._vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._smiles_list)

    @classmethod
    def collate_fn(cls, encoded_seqs):
        """
        Takes a list of encoded sequences and turns them into a batch.
        :param encoded_seqs: A list of sequences in one-hot encoded version.
        :return: A pytorch Tensor with the data correctly padded.
        """
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr
