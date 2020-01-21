#!/usr/bin/env python
#  coding=utf-8

"""
Creates a new model from a set of options.
"""

import argparse

import models.model as mm
import models.vocabulary as mv
import utils.chem as uc
import utils.log as ul


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Create a model with the vocabulary extracted from a SMILES file.")

    parser.add_argument("--input-smiles-path", "-i",
                        help=("SMILES to calculate the vocabulary from. The SMILES are taken as-is, \
                            no processing is done."),
                        type=str, required=True)
    parser.add_argument("--output-model-path", "-o", help="Prefix to the output model.", type=str, required=True)
    parser.add_argument("--num-layers", "-l",
                        help="Number of RNN layers of the model [DEFAULT: 3]", type=int, default=3)
    parser.add_argument("--layer-size", "-s",
                        help="Size of each of the RNN layers [DEFAULT: 512]", type=int, default=512)
    parser.add_argument("--embedding-layer-size", "-e",
                        help="Size of the embedding layer [DEFAULT: 256]", type=int, default=256)
    parser.add_argument("--dropout", "-d",
                        help="Amount of dropout between the GRU layers [DEFAULT: 0.0]", type=float, default=0)
    parser.add_argument("--layer-normalization", "--ln",
                        help="Add layer normalization to the GRU output", action="store_true", default=False)
    parser.add_argument("--max-sequence-length",
                        help="Maximum length of the sequences [DEFAULT: 256]", type=int, default=256)

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    smiles_list = uc.read_smi_file(args.input_smiles_path)

    LOG.info("Building vocabulary")
    tokenizer = mv.SMILESTokenizer()
    vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer)

    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)
    network_params = {
        'num_layers': args.num_layers,
        'num_dimensions': args.layer_size,
        'embedding_layer_size': args.embedding_layer_size,
        'dropout': args.dropout,
        'vocabulary_size': len(vocabulary)
    }
    model = mm.Model(no_cuda=True, vocabulary=vocabulary, tokenizer=tokenizer,
                     network_params=network_params, max_sequence_length=args.max_sequence_length)
    LOG.info("Saving model at %s", args.output_model_path)
    model.save(args.output_model_path)


LOG = ul.get_logger(name="create_model")
if __name__ == "__main__":
    main()
