Implementation of the molecular generative model using randomized SMILES strings
==============================================================================

>
> **Note 1:** The version published alongside [Randomized SMILES strings improve the quality of molecular generative models](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0393-0) is available in the separate branch [randomized_smiles](https://github.com/undeadpixel/reinvent-randomized/tree/randomized_smiles).
>
> **Note 2:** This repository supersedes [undeadpixel/reinvent-gdb13](https://github.com/undeadpixel/reinvent-gdb13).
>

This repository holds the code to create, train and sample models akin to those described in [Randomized SMILES strings improve the quality of molecular generative models](https://chemrxiv.org/articles/Randomized_SMILES_Strings_Improve_the_Quality_of_Molecular_Generative_Models/8639942) and [SMILES-based deep generative scaffold decorator for de-novo drug design](). This version changes the implementation of the model to use packed sequences and several speed improvements. Also, the support for GRU cells has been dropped.

Specifically, it includes the following:

* Python files in the main folder: Scripts to create, train, sample and calculate NLLs of models.
* `./training_sets`: Training set files (in canonical SMILES).

Requirements
------------

This software has been tested on Linux with Tesla V-100 GPUs. We think it should work with other linux-based setups quite easily. 
The create randomized SMILES script uses Spark 2.4 to parallelize the creation of SMILES. By default it should run in local mode, but maybe further configuration is needed.

Install
-------
A [Conda](https://conda.io/miniconda.html) `environment.yml` is supplied with all the required libraries.

~~~~
$> git clone <repo url>
$> cd <repo folder>
$> conda env create -f environment.yml
$> conda activate reinvent-randomized
(reinvent-randomized) $> ...
~~~~

From here the general usage applies.

General Usage
-------------
Four tools are supplied. Further information about the tool's arguments, please run it with `-h`. All output files are in tsv format (the separator is \t).

1) Create Model (`create_model.py`): Creates a blank model file.
2) Train Model (`train_model.py`): Trains the model with the specified parameters.
3) Sample Model (`sample_from_model.py`): Samples an already trained model for a given number of SMILES. It also retrieves the log-likelihood in the process.
4) Calculate NLL (`calculate_nlls.py`): Requires as input a SMILES list and outputs a SMILES list with the NLL calculated for each one. It's recommended not to use files with more than 20-30 million SMILES.
5) Create random SMILES (`create_randomized_smiles.py`): From a list of canonical SMILES it creates a given number of randomized SMILES files and stores them in the folder specified as output with filenames 000.smi, 001.smi, etc.

Usage examples
--------------

Create, train 100 epochs with adaptative learning rate and sample a model with the ChEMBL dataset (randomized SMILES).
~~~~
(reinvent-randomized) $> mkdir -p chembl_randomized/models
(reinvent-randomized) $> ./create_randomized_smiles.py -i training_sets/chembl.training.smi -o chembl_randomized/training -n 100
(reinvent-randomized) $> ./create_randomized_smiles.py -i training_sets/chembl.validation.smi -o chembl_randomized/validation -n 100
(reinvent-randomized) $> ./create_model.py -i chembl_randomized/training/001.smi -o chembl_randomized/models/model.empty
(reinvent-randomized) $> ./train_model.py -i chembl_randomized/models/model.empty -o chembl_randomized/models/model.trained -s chembl_randomized/training -e 100 --lrm ada --csl chembl_randomized/tensorboard --csv chembl_randomized/validation --csn 75000
# (... wait a few days ...)
(reinvent-randomized) $> ./sample_from_model.py -m chembl_randomized/models/model.trained.100 --with-likelihood
~~~~

**CAUTION:** When creating random SMILES sets, the SMILES representation changes and so some of the infrequent tokens do not appear in some sets. To solve that you can try different subsets until you find one that has all the tokens or you can create a fake one with all tokens.

Notice that the tensorboard data is stored in `chembl_randomized/tensorboard` and can be accessed (even during training) by:
~~~~
(reinvent-randomized) $> tensorboard --logdir chembl_randomized/tensorboard --port 9999
~~~~
And go to localhost:9999 to access the web interface.

Create, train 100 epochs with exponential learning rate and sample a model with 1M molecules from the GDB-13 database (canonical SMILES).
~~~~
(reinvent-randomized) $> mkdir -p gdb13_exp/models
(reinvent-randomized) $> ./create_model.py -i training_sets/gdb13.1M.training.smi -o gdb13_exp/models/model.empty
(reinvent-randomized) $> ./train_model.py -i gdb13_exp/models/model.empty -o gdb13_exp/models/model.trained -s training_sets/gdb13.1M.training.smi -e 100 --lrm exp --lrg 0.9 --csl gdb13_exp/tensorboard --csv trained_models/gdb13.1M.validation.smi --csn 10000
# (... wait for some hours ...)
(reinvent-randomized) $> ./sample_from_model.py -m gdb13_exp/models/model.trained.100 --with-likelihood
~~~~

Bugs, Errors, Improvements, etc...
----------------------------------

We have tested the software, but if you find any bug (which there probably are some) don't hesitate to contact us, or even better, send a pull request or open a github issue. If you have any other question, you can contact us at josep.arus@dcb.unibe.ch and we will be happy to answer you :smile:.

