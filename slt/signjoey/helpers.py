# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
from torchtext.legacy.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary


# File and directory management
# Creates a new directory for saving a model
# used in train.py
def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    # directory already exists
    if os.path.isdir(model_dir):
        # raise error
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


# File and directory management
# Logging and configuration handling
# Creates a logger that logs messages to file and console
# Used in train.py, prediction.py
def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    # creates logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        # Writes logs to a file
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux": # Only add console logging on Linux
            # outputs to console
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger


# Logging and configuration handling
# Logs the configuration settings in a structured way
# Used in train.py and test.py
def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict): # If value is a nested dictionary, recurse
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


# Tensor manipulation
# Creates n identical copies of a given PyTorch module.
# Used in transformer_layer.py
def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    # deepcopy() ensures that each copy is independent
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# Tensor manipulation
# Generates a causal mask for Transformer decoding, preventing attention to future tokens
#  used in transformer_layer.py, decoder.py
def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    # creates an upper triangular matrix with ones above the diagonal
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    # == 0 conversion turns it into a boolean mask
    #     True means a token can attend to a position.
    #     False means it cannot attend.
    return torch.from_numpy(mask) == 0


# Sets a fixed random seed for torch, numpy, and random to ensure reproducibility
# Used in train.py
def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Logging and configuration handling
# Logs dataset and vocabulary statistics
#  Used in train.py
def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    # Logs the number of samples in train, validation, and test datasets
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )
    # Logs the first example of gloss (gls) and text (txt)
    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )
    # Logs the first 10 tokens in the gloss and text vocabularies
    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )
    # Logs the vocabulary sizes
    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


# Tokenization processing
# Cleans up Byte Pair Encoding (BPE) output by removing @@ markers
def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    # used in prediction.py (validate_on_data())
    return string.replace("@@ ", "")


# Model checkpoint handling
# Finds the most recent model checkpoint in a directory
# Used in prediction.py (test() function)
def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    # Lists all .ckpt files in a directory
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        # Gets the creation time of a file -> latest creation = latest checkpoint
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


# File and directory management
# Model checkpoint handling
# Loads a saved model checkpoint
#  Used in test.py, train.py
def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


# from onmt
# Tensor manipulation
# Duplicates (tiles) a tensor along a given dimension. Used for beam search
# Used in search.py
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x #If the input is a tuple (like LSTM hidden states (h, c)), it applies tile() to both
        return tile(h, count, dim=dim), tile(c, count, dim=dim)
    # Moves the target dim to the first dimension for processing
    perm = list(range(len(x.size()))) # Get dimensions of x
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0] # Swap target dim to first
        x = x.permute(perm).contiguous()
    out_size = list(x.size()) # Store original shape
    out_size[0] *= count # Multiply size by `count`
    batch = x.size(0)
    # Reshapes the tensor to repeat count times along dim
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0: # Restore original dimensions
        x = x.permute(perm).contiguous()
    return x


# Freezes a module's parameters to prevent updates during training
#  Used in model.py (for freezing embeddings)
def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


# Creates a symbolic link to a target file. If the link already exists, it updates it.
# Used in train.py
def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name) # Create symbolic link
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name) # Remove old link
            os.symlink(target, link_name) # Create new link
        else:
            raise e
