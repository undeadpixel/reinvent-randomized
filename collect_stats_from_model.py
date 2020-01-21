#!/usr/bin/env python
#  coding=utf-8

"""
Collects stats for an existing RNN model.
"""

import argparse

import torch.utils.tensorboard as tbx

import models.model as mm
import models.actions as ma
import utils.log as ul
import utils.chem as uc


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Collects stats from a model.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--training-set-path", "-t",
                        help="Path to the training set SMILES file.", type=str, required=True)
    parser.add_argument("--epoch", "-e", help="Epoch number", type=int, required=True)
    parser = add_stats_args(parser)

    return parser.parse_args()


def add_stats_args(parser, with_prefix=False, with_required=True):  # pylint: disable=missing-docstring
    """
    Adds the args for collect_stats to a parser.
    :param parser: Parser instance.
    :param with_prefix: Add prefix (collect-stats).
    :param with_required: Add required statements where necessary.
    :return: The updated parser
    """
    def _add_arg(name, short_name, help_msg, **kwargs):
        if with_prefix:
            name_arg = "collect-stats-" + name
            short_name_arg = "cs" + short_name
        else:
            name_arg = name
            short_name_arg = short_name
        name_arg = "--" + name_arg
        if len(short_name_arg) > 1:
            short_name_arg = "--" + short_name_arg
        else:
            short_name_arg = "-" + short_name_arg

        required = False
        if "required" in kwargs:
            required = (required or kwargs["required"]) and with_required
            del kwargs["required"]
        parser.add_argument(name_arg, short_name_arg, help=help_msg, required=required, **kwargs)

    _add_arg("log-path", "l", "Path to the log output folder.", type=str, required=True)
    _add_arg("validation-set-path", "v", "Path to the validation set SMILES file.", type=str, required=True)
    _add_arg("sample-size", "n", "Number of SMILES to sample from the model. [DEFAULT: 10000]", type=int, default=10000)
    _add_arg("with-weights", "w", "Store the weight matrices each epoch [DEFAULT: False].",
             action="store_true", default=False)
    _add_arg("smiles-type", "st",
             "SMILES type to converto to TYPES=(smiles, deepsmiles.[branches|rings|both]) [DEFAULT: smiles]",
             type=str, default="smiles")

    return parser


def main():
    """Main function."""
    args = parse_args()

    model = mm.Model.load_from_file(args.model_path, mode="sampling")
    training_set = list(uc.read_smi_file(args.training_set_path))
    validation_set = list(uc.read_smi_file(args.validation_set_path))

    writer = tbx.SummaryWriter(log_dir=args.log_path)

    ma.CollectStatsFromModel(model, args.epoch, training_set, validation_set, writer,
                             sample_size=args.sample_size, with_weights=args.with_weights,
                             to_mol_func=uc.get_mol_func(args.smiles_type), logger=LOG).run()

    writer.close()


if __name__ == "__main__":
    LOG = ul.get_logger("collect_stats_from_model")
    main()
