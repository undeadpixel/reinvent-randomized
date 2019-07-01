#!/usr/bin/env python

import argparse
import os
import functools

import utils.log as ul
import utils.chem as uc
import utils.spark as us


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(
        description="Creates many datasets.")
    parser.add_argument("--input-smi-path", "-i", help="Path to a SMILES file to convert.", type=str, required=True)
    parser.add_argument("--output-smi-folder-path", "-o",
                        help="Path to a folder that will have the converted SMILES files.", type=str, required=True)
    parser.add_argument("--random-type", "-r", help="Type of the converted SMILES TYPES=(restricted,unrestricted) \
        [DEFAULT: restricted].", type=str, default="restricted")
    parser.add_argument(
        "--num-files", "-n", help="Number of SMILES files to create (numbered from 000 ...) [DEFAULT: 1]",
        type=int, default=1)
    parser.add_argument("--num-partitions", "-p", help="Number of SPARK partitions to use [DEFAULT: 1000]",
                        type=int, default=1000)

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    mols_rdd = SC.textFile(args.input_smi_path) \
        .repartition(args.num_partitions) \
        .map(uc.to_mol)\
        .persist()

    os.makedirs(args.output_smi_folder_path, exist_ok=True)

    smiles_func = functools.partial(uc.randomize_smiles, random_type=args.random_type)
    for i in range(args.num_files):
        with open("{}/{:03d}.smi".format(args.output_smi_folder_path, i), "w+") as out_file:
            for smi in mols_rdd.map(smiles_func).collect():
                out_file.write("{}\n".format(smi))

    mols_rdd.unpersist()


LOG = ul.get_logger("create_randomized_smiles")
if __name__ == "__main__":
    SPARK, SC = us.SparkSessionSingleton.get("create_randomized_smiles")
    main()
