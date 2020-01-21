# coding=utf-8

"""
Implementation of the RNN model
"""

import torch
import torch.nn as tnn
import torch.nn.utils.rnn as tnnur

import models.vocabulary as mv


class RNN(tnn.Module):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, vocabulary_size, num_dimensions, num_layers, embedding_layer_size, dropout):
        """
        Implements a N layer GRU|LSTM cell including an embedding layer and an output linear layer
        back to the size of the vocabulary.
        :param voc_size: Size of the vocabulary.
        :param num_dimensions: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param cell_type: Cell type to use (GRU or LSTM).
        :param embedding_layer_size: Size of the embedding layer.
        :param dropout: Dropout to add between cell layers.
        :return:
        """
        super(RNN, self).__init__()

        self.num_dimensions = num_dimensions
        self.embedding_layer_size = embedding_layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocabulary_size = vocabulary_size

        self._embedding = tnn.Sequential(
            tnn.Embedding(self.vocabulary_size, self.embedding_layer_size),
            tnn.Dropout(self.dropout)
        )
        self._rnn = tnn.LSTM(self.embedding_layer_size, self.num_dimensions,
                             num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self._linear = tnn.Linear(self.num_dimensions, self.vocabulary_size)

    def forward(self, padded_seqs, seq_lengths, hidden_state=None):  # pylint: disable=W0221
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param padded_seqs: Padded input tensor (batch_size, seq_size).
        :param seq_lengths: Length of each sequence in the batch.
        :param hidden_state: Hidden state tensor.
        :return: A tuple with the output state and the output hidden state.
        """
        batch_size = padded_seqs.size(0)
        if hidden_state is None:
            size = (self.num_layers, batch_size, self.num_dimensions)
            hidden_state = [torch.zeros(*size).cuda(), torch.zeros(*size).cuda()]

        padded_encoded_seqs = self._embedding(padded_seqs)  # (batch,seq,embedding)
        packed_encoded_seqs = tnnur.pack_padded_sequence(
            padded_encoded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_encoded_seqs, hidden_state = self._rnn(packed_encoded_seqs, hidden_state)
        padded_encoded_seqs, _ = tnnur.pad_packed_sequence(packed_encoded_seqs, batch_first=True)

        mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        logits = self._linear(padded_encoded_seqs)*mask
        return (logits, hidden_state)

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        :return: A dict with the params of the model.
        """
        return {
            'dropout': self.dropout,
            'num_dimensions': self.num_dimensions,
            'num_layers': self.num_layers,
            'embedding_layer_size': self.embedding_layer_size,
            'vocabulary_size': self.vocabulary_size
        }


class Model:
    """
    Implements an RNN model using SMILES.
    """

    def __init__(self, vocabulary, tokenizer, network_params=None, max_sequence_length=256, no_cuda=False,
                 mode="train"):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Network params to initialize the RNN.
        :param max_sequence_length: Sequences longer than this value will not be processed.
        :param no_cuda: The model is explicitly initialized as not using cuda, even if cuda is available.
        :param mode: Training or eval mode.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        self.network = RNN(**network_params)
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()

        self.nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

        self.set_mode(mode)

    @classmethod
    def load_from_file(cls, file_path, mode="train"):
        """
        Loads a model from a single file
        :param file_path: Path of the file where the model data was previously stored.
        :param mode: Mode to load the model as (training or eval).
        :return: A new instance of the Model or an exception if it was not possible to load it.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

        network_params = save_dict.get("network_params", {})
        model = Model(
            vocabulary=save_dict['vocabulary'],
            tokenizer=save_dict.get('tokenizer', mv.SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict['max_sequence_length'],
            mode=mode
        )
        model.network.load_state_dict(save_dict["network"])
        return model

    def set_mode(self, mode):
        """
        Changes the mode of the RNN to training or eval.
        :param mode: Mode to change to (training, eval)
        :return: The model instance.
        """
        if mode == "sampling" or mode == "eval":
            self.network.eval()
        else:
            self.network.train()
        return self

    def save(self, path):
        """
        Saves the model to a file.
        :param path: Path to save the model to.
        """
        save_dict = {
            'vocabulary': self.vocabulary,
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_sequence_length,
            'network': self.network.state_dict(),
            'network_params': self.network.get_params()
        }
        torch.save(save_dict, path)

    def likelihood(self, padded_seqs, seq_lengths):
        """
        Retrieves the likelihood of a given sequence. Used in training.
        :param padded_seqs: (batch_size, sequence_length) A batch of padded sequences.
        :param seq_lengths: Length of each sequence in a tensor.
        :return:  (batch_size) Log likelihood for each example.
        """
        logits, _ = self.network(padded_seqs, seq_lengths - 1)
        log_probs = logits.log_softmax(dim=2).transpose(1, 2)
        return self.nll_loss(log_probs, padded_seqs[:, 1:]).sum(dim=1)

    @torch.no_grad()
    def sample_smiles(self, num):
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :return: An iterator with (smiles, likelihood) pairs
        """
        input_vector = torch.full((num, 1), self.vocabulary["^"], dtype=torch.long).cuda()  # (batch, 1)
        seq_lengths = torch.ones(num).cuda()  # (batch)
        sequences = []
        hidden_state = None
        nlls = torch.zeros(num).cuda()
        not_finished = torch.ones(num, 1, dtype=torch.long).cuda()
        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector, seq_lengths, hidden_state)  # (batch, 1, voc)
            probs = logits.softmax(dim=2).squeeze()  # (batch, voc)
            log_probs = logits.log_softmax(dim=2).squeeze()
            input_vector = torch.multinomial(probs, 1)*not_finished  # (batch, 1)
            sequences.append(input_vector)
            nlls += self.nll_loss(log_probs, input_vector.squeeze())
            not_finished = (input_vector > 1).type(torch.long)
            if not_finished.sum() == 0:
                break

        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq))
                  for seq in torch.cat(sequences, 1).data.cpu().numpy()]
        nlls = nlls.data.cpu().numpy().tolist()
        return zip(smiles, nlls)
