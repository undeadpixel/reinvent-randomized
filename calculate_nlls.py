#!/usr/bin/env python
#  coding=utf-8

"""
Calculates the NLLs of a set of molecules.
"""

import argparse

import models.model as mm
import models.actions as ma

import utils.log as ul
import utils.chem as uc


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Calculates NLLs of a list of molecules given a model.")
    parser.add_argument("--input-csv-path", "-i",
                        help="Path to the input CSV file. The first field should be SMILES strings and the rest are \
                            going to be kept as-is.",
                        type=str, required=True)
    parser.add_argument("--output-csv-path", "-o",
                        help="Path to the output CSV file which will have the NLL added as a new field in the end.",
                        type=str, required=True)
    parser.add_argument("--model-path", "-m", help="Path to the model that will be used.", type=str, required=True)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size used to calculate NLLs (DEFAULT: 128).", type=int, default=128)
    parser.add_argument("--use-gzip", help="Compress the output file (if set).", action="store_true", default=False)

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    model = mm.Model.load_from_file(args.model_path, mode="sampling")

    input_csv = uc.open_file(args.input_csv_path, mode="rt")
    if args.use_gzip:
        args.output_csv_path += ".gz"
    output_csv = uc.open_file(args.output_csv_path, mode="wt+")

    calc_nlls_action = ma.CalculateNLLsFromModel(model, batch_size=args.batch_size, logger=LOG)
    smiles_list = list(uc.read_smi_file(args.input_csv_path))

    for nll in ul.progress_bar(calc_nlls_action.run(smiles_list), total=len(smiles_list)):
        input_line = input_csv.readline().strip()
        output_csv.write("{}\t{:.8f}\n".format(input_line, nll))

    input_csv.close()
    output_csv.close()


LOG = ul.get_logger("calculate_nlls")
if __name__ == "__main__":
    main()
