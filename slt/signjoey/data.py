# coding: utf-8
"""
Data module
"""
import os
import sys
import random
from xml.etree.ElementInclude import include

import torch
try:
    from torchtext.legacy import data # Handles dataset creation, tokenization, and batching
    from torchtext.legacy.data import Dataset, Iterator
except ModuleNotFoundError:
    from torchtext import data # Handles dataset creation, tokenization, and batching
    from torchtext.data import Dataset, Iterator
from torchvision.io import read_image,read_video # Reads image/video frames
from torchvision import transforms as t
import socket
import math
from signjoey.dataset import SignTranslationDataset #Custom dataset class for SLT
# Handles vocabulary construction for glosses & text.
from signjoey.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)


def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    # Extracts dataset paths from the data_cfg dictionary.
    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
        # dimensionality of the input sign features
        pad_feature_size = sum(data_cfg["feature_size"])

    else:
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths = os.path.join(data_path, data_cfg["dev"])
        test_paths = os.path.join(data_path, data_cfg["test"])
        pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]
    include_masks = data_cfg.get("include_masks",False)
    print("Include masks is :", include_masks)

    # Tokenizes text sequences based on character (char) or word (word) level
    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()

    # placeholder?
    def tokenize_features(features):
        # ft_list = torch.split(features, 1, dim=0)
        # return [ft.squeeze() for ft in ft_list]
        return features

    # NOTE (Cihan): The something was necessary to match the function signature.
    # Stacks video frames to match batch processing requirements
    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    def load_rgb_frames(batch, something):
        features = []
        missing = 0
        count = 0
        for filenames in batch:
            frames = []
            
            for filename in filenames:
                count +=1
                
                
                try:
                    # read rgb frames
                    img = read_image(filename).unsqueeze(0)
                    # resise to 200x200
                    img = t.functional.resize(img,[200,200])
                    # if count ==1:
                    #     print(img)
                    # Normalize to [-1, 1]
                    img = (img / 255.) * 2 - 1
                    
                    frames.append(img)
                # If an image fails to load, it replaces it with a zero-filled tensor
                except Exception as e:
                    if isinstance(filename, str):

                        print("image did not load", filename)
                    frames.append((torch.zeros(1,3,200,200)/255.)* 2 - 1)
            
            frames = torch.cat(frames,dim=0)
            frames = frames.permute(1,0,2,3).float()
            
            features.append(frames)
        

        return torch.stack(features, dim=0)

    # Creating Masking for Variable-Length Sequences
    def get_src_mask(batch, something):
        lengths = []
        masks = []
        # Computes source sequence lengths from filenames.
        for filenames in batch:
            og_length = len(list(filter(lambda x: isinstance(x,str),filenames)))
            if (og_length%8 > 4):
                lengths.append(math.floor(og_length/8)+1)
            else:
                lengths.append(math.floor(og_length/8))
            # lengths.append(math.ceil(og_length))
            # lengths.append(math.floor(og_length/8)-1)
            # lengths.append(math.ceil(len(filenames)/4))
            
            
        max_len = max(lengths)
        # Creates a binary mask (1 for real tokens, 0 for padding)
        # Helps the model ignore padding during attention
        for length in lengths:
            mask = torch.cat([torch.ones(length),torch.zeros(max_len - length)])  
            masks.append(mask)     
        
        return torch.stack(masks, dim=0)

    # Stores raw metadata
    sequence_field = data.RawField()
    signer_field = data.RawField()

    # Stores sign language video features (not tokenized)
    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        preprocessing=tokenize_features,
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        # postprocessing=load_rgb_frames,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    if include_masks:
        mask_field = data.Field(
            use_vocab=False,
            init_token=None,
            dtype=torch.float32,
            preprocessing=tokenize_features,
            tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
            batch_first=True,
            include_lengths=False,
            postprocessing=get_src_mask,
            pad_token=torch.zeros((pad_feature_size,)),
        )

    # # Stores gloss sequences
    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    # Stores spoken text sequences
    txt_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    # Loads sign language translation dataset.
    if include_masks:
        train_data = SignTranslationDataset(
            path=train_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field,mask_field),
            # Filters out long sequences
            filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
            and len(vars(x)["txt"]) <= max_sent_length,
            include_masks=include_masks,
        )

    else:
        train_data = SignTranslationDataset(
            path=train_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
            filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
            and len(vars(x)["txt"]) <= max_sent_length,
        )

    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    # Builds vocabulary for glosses and text
    gls_vocab = build_vocab(
        field="gls",
        # ensures only frequent words are included
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
        vocab_file=gls_vocab_file,
    )
    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
        vocab_file=txt_vocab_file,
    )
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep

    # Handling Development and Test Datasets
    if include_masks:
        dev_data = SignTranslationDataset(
            path=dev_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field, mask_field),
            include_masks=include_masks
        )
    else:
        dev_data = SignTranslationDataset(
            path=dev_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
        )
    # selects a random subset of the development data
    # Helps reduce computation time during early experimentation
    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        # select this many development examples randomly and discard the rest
        keep_ratio = random_dev_subset / len(dev_data)
        keep, _ = dev_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        dev_data = keep

    # check if target exists
    # Loads test dataset, just like train_data and dev_data
    if include_masks:
        test_data = SignTranslationDataset(
            path=test_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field,mask_field),
            include_masks=include_masks
        )
    else:
        test_data = SignTranslationDataset(
            path=test_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
        )

    # Ensures that gloss and text tokens are mapped correctly
    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab


# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
# Adjusts batch sizes dynamically based on sequence length
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


# Creates an iterator to load data efficiently
def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    #     "sentence" → Batch size is number of sentences
    #     "token" → Batch size is total number of tokens.
    batch_type: str = "sentence",
    #     train=True → Enables sorting & shuffling.
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
    
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            # Sorts batches by sequence length
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter
