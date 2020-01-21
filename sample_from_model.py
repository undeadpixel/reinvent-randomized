#!/usr/bin/env python
#  coding=utf-8

"""
Samples an existing RNN model.
"""

import argparse
import gzip
import functools

import tqdm

import models.model as mm
import models.actions as ma
import utils.log as ul


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Samples a model.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--output-smiles-path", "-o",
                        help="Path to the output file (if none given it will use stdout).", type=str)
    parser.add_argument("--num", "-n", help="Number of SMILES to sample [DEFAULT: 1024]", type=int, default=1024)
    parser.add_argument("--with-nll", help="Store the NLL in a column after the SMILES.",
                        action="store_true", default=False)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size (beware GPU memory usage) [DEFAULT: 128]", type=int, default=128)
    parser.add_argument("--use-gzip", help="Compress the output file (if set).", action="store_true", default=False)

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    model = mm.Model.load_from_file(args.model_path, mode="eval")

    open_func = open
    if args.use_gzip:
        open_func = gzip.open
        args.output_smiles_path += ".gz"

    if args.output_smiles_path:
        csv_file = open_func(args.output_smiles_path, "wt+")
        write_func = functools.partial(csv_file.write)
    else:
        csv_file = tqdm.tqdm
        write_func = functools.partial(csv_file.write, end="")

    sample_model = ma.SampleModel(model, args.batch_size)

    for smi, nll in ul.progress_bar(sample_model.run(args.num), total=args.num):
        output_row = [smi]
        if args.with_nll:
            output_row.append("{:.8f}".format(nll))
        write_func("\t".join(output_row) + "\n")

    if args.output_smiles_path:
        csv_file.close()


LOG = ul.get_logger(name="sample_from_model")
if __name__ == "__main__":
    main()
