# coding=utf-8

"""
Implementation of the RNN model
"""

import torch
import torch.nn as tnn

import models.vocabulary as mv


class RNN(tnn.Module):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, layer_size=512, num_layers=3, cell_type='gru', embedding_layer_size=256,
                 dropout=0.):
        """
        Implements a N layer GRU|LSTM cell including an embedding layer and an output linear layer
        back to the size of the vocabulary.
        :param voc_size: Size of the vocabulary.
        :param layer_size: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param cell_type: Cell type to use (GRU or LSTM).
        :param embedding_layer_size: Size of the embedding layer.
        :param dropout: Dropout to add between cell layers.
        :return:
        """
        super(RNN, self).__init__()

        self.layer_size = layer_size
        self.embedding_layer_size = embedding_layer_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.dropout = dropout

        self._embedding = tnn.Embedding(voc_size, self.embedding_layer_size)
        if self.cell_type == 'gru':
            self._rnn = tnn.GRU(self.embedding_layer_size, self.layer_size, num_layers=self.num_layers,
                                dropout=self.dropout, batch_first=True)
        elif self.cell_type == 'lstm':
            self._rnn = tnn.LSTM(self.embedding_layer_size, self.layer_size, num_layers=self.num_layers,
                                 dropout=self.dropout, batch_first=True)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')
        self._linear = tnn.Linear(self.layer_size, voc_size)

    def forward(self, input_vector, hidden_state=None):  # pylint: disable=W0221
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor.
        :return: A tuple with the output state and the output hidden state.
        """
        batch_size, seq_size = input_vector.size()
        if hidden_state is None:
            size = (self.num_layers, batch_size, self.layer_size)
            if self.cell_type == "gru":
                hidden_state = torch.zeros(*size)
            else:
                hidden_state = [torch.zeros(*size), torch.zeros(*size)]
        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)

        output_vector = output_vector.reshape(-1, self.layer_size)

        output_data = self._linear(output_vector).view(batch_size, seq_size, -1)

        return output_data, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        :return: A dict with the params of the model.
        """
        return {
            'dropout': self.dropout,
            'layer_size': self.layer_size,
            'num_layers': self.num_layers,
            'cell_type': self.cell_type,
            'embedding_layer_size': self.embedding_layer_size
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

        self.network = RNN(len(self.vocabulary), **network_params)
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()

        self.nll_loss = tnn.NLLLoss(reduction="none")

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

    def likelihood(self, sequences):
        """
        Retrieves the likelihood of a given sequence. Used in training.
        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """
        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)
        return self.nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    @torch.no_grad()
    def sample_smiles(self, num):
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :return: An iterator with (smiles, likelihood) pairs
        """
        start_token = torch.zeros(num, dtype=torch.long)
        start_token[:] = self.vocabulary["^"]
        input_vector = start_token
        sequences = []

        hidden_state = None
        nlls = torch.zeros(num)
        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)

            input_vector = torch.multinomial(probabilities, 1).view(-1)
            sequences.append(input_vector.view(-1, 1))
            nlls += self.nll_loss(log_probs, input_vector)
            if input_vector.sum() == 0:
                break

        sequences = torch.cat(sequences, 1)
        nlls = nlls.data.cpu().numpy().tolist()
        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in sequences.data.cpu().numpy()]
        return zip(smiles, nlls)
