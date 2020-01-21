"""
RDKit util functions.
"""
import random
import gzip
import re
import functools

import rdkit.Chem as rkc
import deepsmiles as ds


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()


def read_smi_file(file_path, ignore_invalid=True, num=-1):
    """
    Reads a SMILES file.
    :param file_path: Path to a SMILES file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num SMILES.
    :return: A list with all the SMILES.
    """
    return map(lambda fields: fields[0], read_csv_file(file_path, ignore_invalid, num))


def read_csv_file(file_path, ignore_invalid=True, num=-1):
    """
    Reads a SMILES file.
    :param file_path: Path to a CSV file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num rows.
    :return: An iterator with the rows.
    """
    with open_file(file_path, "rt") as csv_file:
        for i, row in enumerate(csv_file):
            if i == num:
                break
            fields = row.rstrip().split("\t")
            if fields:
                yield fields
            elif not ignore_invalid:
                yield None


def open_file(path, mode="r", with_gzip=False):
    """
    Opens a file depending on whether it has or not gzip.
    :param path: Path where the file is located.
    :param mode: Mode to open the file.
    :param with_gzip: Open as a gzip file anyway.
    """
    open_func = open
    if path.endswith(".gz") or with_gzip:
        open_func = gzip.open
    return open_func(path, mode)


def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)


def to_smiles(mol):
    """
    Converts a Mol object into a canonical SMILES string.
    :param mol: Mol object.
    :return: A SMILES string.
    """
    return rkc.MolToSmiles(mol, isomericSmiles=False)


def randomize_smiles(mol, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None

    if random_type == "unrestricted":
        return rkc.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
        return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))


DEEPSMI_CONVERTERS = {
    "rings": ds.Converter(rings=True),
    "branches": ds.Converter(branches=True),
    "both": ds.Converter(rings=True, branches=True)
}


def to_deepsmiles(smi, converter="both"):
    """
    Converts a SMILES strings to the DeepSMILES alternative.
    :param smi: SMILES string.
    :return : A DeepSMILES string.
    """
    return DEEPSMI_CONVERTERS[converter].encode(smi)


def from_deepsmiles(deepsmi, converter="both"):
    """
    Converts a DeepSMILES strings to the SMILES alternative.
    :param smi: DeepSMILES string.
    :return : A SMILES string or None if it's invalid
    """
    try:
        return DEEPSMI_CONVERTERS[converter].decode(deepsmi)
    except:  # pylint:disable=bare-except
        return None


def get_mol_func(smiles_type):
    """
    Returns a function pointer that converts a given SMILES type to a mol object.
    :param smiles_type: The SMILES type to convert VALUES=(deepsmiles.*, smiles, scaffold).
    :return : A function pointer.
    """

    if smiles_type.startswith("deepsmiles"):
        _, deepsmiles_type = smiles_type.split(".")
        return lambda deepsmi: to_mol(from_deepsmiles(deepsmi, converter=deepsmiles_type))
    else:
        return to_mol


def get_smi_func(smiles_type):
    """
    Returns a function pointer that converts a given SMILES string to SMILES of the given type.
    :param smiles_type: The SMILES type to convert VALUES=(deepsmiles.*, smiles, scaffold).
    :return : A function pointer.
    """
    if smiles_type.startswith("deepsmiles"):
        _, deepsmiles_type = smiles_type.split(".")
        return functools.partial(to_deepsmiles, converter=deepsmiles_type)
    elif smiles_type == "scaffold":
        return add_brackets_to_attachment_points
    else:
        return lambda x: x


ATTACHMENT_POINT_TOKEN = "*"
ATTACHMENT_POINT_NO_BRACKETS_REGEXP = r"(?<!\[){}".format(re.escape(ATTACHMENT_POINT_TOKEN))


def add_brackets_to_attachment_points(smi):
    """
    Adds brackets to the attachment points (if they don't have them).
    :param smi: SMILES string.
    :return: A SMILES string with attachments with brackets.
    """
    return re.sub(ATTACHMENT_POINT_NO_BRACKETS_REGEXP, "[{}]".format(ATTACHMENT_POINT_TOKEN), smi)
