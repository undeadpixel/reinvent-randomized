import random
import math

import numpy as np
import scipy.stats as sps

import torch
import torch.utils.data as tud
import torch.nn.utils as tnnu

import models.dataset as md
import models.vocabulary as mv
import utils.chem as uc
import utils.tensorboard as utb


class Action:
    def __init__(self, logger=None):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self.logger = logger

    def _log(self, level, msg, *args):
        """
        Logs a message with the class logger.
        :param level: Log level.
        :param msg: Message to log.
        :param *args: The arguments to escape.
        :return:
        """
        if self.logger:
            getattr(self.logger, level)(msg, *args)


class TrainModelPostEpochHook(Action):

    def __init__(self, logger=None):
        """
        Initializes a training hook that runs after every epoch.
        This hook enables to save the model, change LR, etc. during training.
        :return:
        """
        Action.__init__(self, logger)

    def run(self, model, training_set, epoch):  # pylint: disable=unused-argument
        """
        Performs the post-epoch hook. Notice that model should be modified in-place.
        :param model: Model instance trained up to that epoch.
        :param training_set: List of SMILES used as the training set.
        :param epoch: Epoch number (for logging purposes).
        :return: Boolean that indicates whether the training should continue or not.
        """
        return True  # simply does nothing...


class TrainModel(Action):

    def __init__(self, model, optimizer, training_sets, batch_size, clip_gradient,
                 epochs, post_epoch_hook=None, logger=None):
        """
        Initializes the training of an epoch.
        : param model: A model instance, not loaded in sampling mode.
        : param optimizer: The optimizer instance already initialized on the model.
        : param training_set: A list with the training set SMILES, either cycled using \
            itertools.cycle or as many as epochs needed to train.
        : param batch_size: Batch size to use.
        : param clip_gradient: Clip the gradients after each backpropagation.
        : return:
        """
        Action.__init__(self, logger)

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.clip_gradient = clip_gradient
        self.batch_size = batch_size
        self.training_sets = training_sets

        if not post_epoch_hook:
            self.post_epoch_hook = TrainModelPostEpochHook(logger=self.logger)
        else:
            self.post_epoch_hook = post_epoch_hook

    def run(self):
        """
        Performs a training epoch with the parameters used in the constructor.
        :return: An iterator of (total_batches, epoch_iterator), where the epoch iterator
                  returns the loss function at each batch in the epoch.
        """
        for epoch, training_set in zip(range(1, self.epochs + 1), self.training_sets):
            dataloader = self._initialize_dataloader(training_set)
            epoch_iterator = self._epoch_iterator(dataloader)
            yield len(dataloader), epoch_iterator

            self.model.set_mode("eval")
            post_epoch_status = self.post_epoch_hook.run(self.model, training_set, epoch)
            self.model.set_mode("train")

            if not post_epoch_status:
                break

    def _epoch_iterator(self, dataloader):
        for padded_seqs, seq_lengths in dataloader:
            loss = self.model.likelihood(padded_seqs, seq_lengths).mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient > 0:
                tnnu.clip_grad_norm_(self.model.network.parameters(), self.clip_gradient)

            self.optimizer.step()

            yield loss

    def _initialize_dataloader(self, training_set):
        dataset = md.Dataset(smiles_list=training_set, vocabulary=self.model.vocabulary, tokenizer=mv.SMILESTokenizer())
        return tud.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              collate_fn=md.Dataset.collate_fn)


class CollectStatsFromModel(Action):
    """Collects stats from an existing RNN model."""

    def __init__(self, model, epoch, training_set, validation_set, writer, sample_size,
                 with_weights=False, to_mol_func=uc.to_mol, other_values=None, logger=None):
        """
        Creates an instance of CollectStatsFromModel.
        : param model: A model instance initialized as sampling_mode.
        : param epoch: Epoch number to be sampled(informative purposes).
        : param training_set: Iterator with the training set.
        : param validation_set: Iterator with the validation set.
        : param writer: Writer object(Tensorboard writer).
        : param other_values: Other values to save for the epoch.
        : param sample_size: Number of molecules to sample from the training / validation / sample set.
        : param with_weights: To calculate or not the weights.
        : param to_mol_func: Mol function used(change for deepsmiles or other representations).
        : return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.epoch = epoch
        self.training_set = training_set
        self.validation_set = validation_set
        self.writer = writer
        self.other_values = other_values

        self.with_weights = with_weights
        self.sample_size = max(sample_size, 1)
        self.to_mol_func = to_mol_func

        self.data = {}

        self._calc_nlls_action = CalculateNLLsFromModel(self.model, 128, self.logger)

    @torch.no_grad()
    def run(self):
        """
        Collects stats for a specific model object, epoch, validation set, training set and writer object.
        : return: A dictionary with all the data saved for that given epoch.
        """
        self._log("info", "Collecting data for epoch %s", self.epoch)
        self.data = {}

        self._log("debug", "Sampling SMILES")
        sampled_smis, sampled_nlls = [np.array(a) for a in zip(*self.model.sample_smiles(num=self.sample_size))]

        self._log("debug", "Obtaining molecules from SMILES")
        sampled_mols = [smi_mol for smi_mol in [(smi, self.to_mol_func(smi)) for smi in sampled_smis] if smi_mol[1]]

        self._log("debug", "Calculating NLLs for the validation and training sets")
        validation_nlls, training_nlls = self._calculate_validation_training_nlls()

        if self.with_weights:
            self._log("debug", "Calculating weight stats")
            self._weight_stats()

        self._log("debug", "Calculating nll stats")
        self._nll_stats(sampled_nlls, validation_nlls, training_nlls)

        self._log("debug", "Calculating validity stats")
        self._valid_stats(sampled_mols)

        self._log("debug", "Drawing some molecules")
        self._draw_mols(sampled_mols)

        if self.other_values:
            self._log("debug", "Adding other values")
            for name, val in self.other_values.items():
                self._add_scalar(name, val)

        return self.data

    def _calculate_validation_training_nlls(self):
        def calc_nlls(smiles_set):
            subset = random.sample(smiles_set, self.sample_size)
            return np.array(list(self._calc_nlls_action.run(subset)))

        return (calc_nlls(self.validation_set), calc_nlls(self.training_set))

    def _valid_stats(self, mols):
        self._add_scalar("valid", 100.0*len(mols)/self.sample_size)

    def _weight_stats(self):
        for name, weights in self.model.network.named_parameters():
            self._add_histogram("weights/{}".format(name), weights.clone().cpu().data.numpy())

    def _nll_stats(self, sampled_nlls, validation_nlls, training_nlls):
        self._add_histogram("nll_plot/sampled", sampled_nlls)
        self._add_histogram("nll_plot/validation", validation_nlls)
        self._add_histogram("nll_plot/training", training_nlls)

        self._add_scalars("nll/avg", {
            "sampled": sampled_nlls.mean(),
            "validation": validation_nlls.mean(),
            "training": training_nlls.mean()
        })

        self._add_scalars("nll/var", {
            "sampled": sampled_nlls.var(),
            "validation": validation_nlls.var(),
            "training": training_nlls.var()
        })

        def jsd(dists):
            min_size = min(len(dist) for dist in dists)
            dists = [dist[:min_size] for dist in dists]
            num_dists = len(dists)
            avg_dist = np.sum(dists, axis=0) / num_dists
            return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists

        self._add_scalar("nll_plot/jsd_joined", jsd([sampled_nlls, training_nlls, validation_nlls]))

    def _draw_mols(self, mols):
        try:
            smis, mols = zip(*random.sample(mols, 20))
            utb.add_mols(self.writer, "molecules", mols, mols_per_row=4, legends=smis, global_step=self.epoch)
        except ValueError:
            pass

    def _add_scalar(self, key, val):
        self.data[key] = val
        self.writer.add_scalar(key, val, self.epoch)

    def _add_scalars(self, key, dict_vals):
        for k, val in dict_vals.items():
            self.data["{}.{}".format(key, k)] = val
        self.writer.add_scalars(key, dict_vals, self.epoch)

    def _add_histogram(self, key, vals):
        self.data[key] = vals
        self.writer.add_histogram(key, vals, self.epoch)


class SampleModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in sampling mode).
        :params batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, num):
        """
        Samples the model for the given number of SMILES.
        :params num: Number of SMILES to sample.
        :return: An iterator with each of the batches sampled in (smiles, nll) pairs.
        """
        num_batches = math.ceil(num / self.batch_size)
        molecules_left = num
        for _ in range(num_batches):
            current_batch_size = min(molecules_left, self.batch_size)
            for smi, nll in self.model.sample_smiles(current_batch_size):
                yield (smi, nll)
            molecules_left -= current_batch_size


class CalculateNLLsFromModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A model instance.
        :param batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, smiles_list):
        """
        Calculates the NLL for a set of SMILES strings.
        :param smiles_list: List with SMILES.
        :return: An iterator with each NLLs in the same order as the SMILES list.
        """
        dataset = md.Dataset(smiles_list, self.model.vocabulary, self.model.tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size, collate_fn=md.Dataset.collate_fn,
                                    shuffle=False)
        for batch in dataloader:
            for nll in self.model.likelihood(*batch).data.cpu().numpy():
                yield nll
