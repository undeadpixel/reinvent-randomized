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
    parser.add_argument("--smiles-type", "-s", help="Type of SMILES strings TYPES=(smiles,deepsmiles.*,scaffold) \
        [DEFAULT: smiles].", type=str, default="restricted")
    parser.add_argument(
        "--num-files", "-n", help="Number of SMILES files to create (numbered from 000 ...) [DEFAULT: 1]",
        type=int, default=1)
    parser.add_argument("--num-partitions", "-p", help="Number of SPARK partitions to use [DEFAULT: 1000]",
                        type=int, default=1000)

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    randomize_func = functools.partial(uc.randomize_smiles, random_type=args.random_type)
    to_mol_func = uc.get_mol_func(args.smiles_type)
    to_smiles_func = uc.get_smi_func(args.smiles_type)
    mols_rdd = SC.textFile(args.input_smi_path) \
        .repartition(args.num_partitions) \
        .map(to_mol_func)\
        .persist()

    os.makedirs(args.output_smi_folder_path, exist_ok=True)

    for i in range(args.num_files):
        with open("{}/{:03d}.smi".format(args.output_smi_folder_path, i), "w+") as out_file:
            for smi in mols_rdd.map(lambda mol: to_smiles_func(randomize_func(mol))).collect():
                out_file.write("{}\n".format(smi))


LOG = ul.get_logger("create_randomized_smiles")
if __name__ == "__main__":
    SPARK, SC = us.SparkSessionSingleton.get("create_randomized_smiles")
    main()
