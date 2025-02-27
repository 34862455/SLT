# coding: utf-8
"""
Data module
"""
from torchtext.legacy import data
from torchtext.legacy.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import re

#  Used in SignTranslationDataset()
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f: #decompress the file
        loaded_object = pickle.load(f) #deserialize the dataset object
        return loaded_object

# Processing sign language frames
# Generates paths for individual sign language frames
#  Used in SignTranslationDataset()
def list_frames(image_dir,direct, filenames):
    # Initializes an empty list for storing frame paths
    frames = []
    missing = 0
    count = 0
    #print(len(filenames))
    # Iterates over filenames and constructs the full path for each frame
    for filename in filenames:
        
        if filename:
            count +=1
            try:
                #img = read_image(os.path.join(image_dir,direct,filenames[i])).unsqueeze(0)
                # Appends the full path as a string to frames
                frames.append("{0}/{1}/{2}".format("/media/botlhale/My Book/Datasets/SASL Corpus png",direct,filename))
                
            except:
                #print(filename)
                missing +=1 #If an image is missing
    
    # frames = torch.cat(frames,dim=0)
    # frames = t.functional.resize(frames,[180,180])
    # frames = frames.permute(1,0,2,3)
    #print("shape", frames.shape)
    #print(len(frames))
    return frames

# Creating a structured dataset
# Creates a torchtext dataset for Sign Language Translation
# Used in data.py (in load_data() function)
class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    # Sorts examples based on sign (sgn) and text (txt) lengths
    # Helps efficient batching in torchtext
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    # Loads, processes, and structures dataset
    def __init__(
        self,
        path: str, #Path(s) to dataset files
        fields: Tuple[RawField, RawField, Field, Field, Field], #Tuple of torchtext fields
        include_masks: bool = False, #Whether to include masks for missing frames
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        self.include_masks = include_masks
        if not isinstance(fields[0], (tuple, list)):
            # Assigns column names based on whether masks are included
            # Feels redundant but okay
            if include_masks:
                fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("sgn", fields[2]),
                    ("gls", fields[3]),
                    ("txt", fields[4]),
                    ("mask",fields[5]),
                ]
            else:
                fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("sgn", fields[2]),
                    ("gls", fields[3]),
                    ("txt", fields[4]),
                ]

        if not isinstance(path, list):
            path = [path] # Convert single path to list

        samples = {}
        for annotation_file in path:
            # call
            tmp = load_dataset_file(annotation_file)
            if include_masks:
                # Extracts frames, gloss, and text
                for files,direct,gloss,transl, img_dir in tmp:
                    seq_id = direct
                    if seq_id in samples:
                        assert samples[seq_id]["name"] == direct
                        # Cleans text
                        assert samples[seq_id]["text"] == re.sub(r'[^\w\s]',' ',transl.replace("'",""))
                        assert samples[seq_id]["gloss"] == gloss
                        
                    else:
                        samples[seq_id] = {
                            "name": direct,
                            "signer": "",
                            "gloss": gloss,
                            # Cleans text
                            "text": re.sub(r'[^\w\s]',' ',transl.replace("'","")),
                            # load frame paths
                            "sign": list_frames(img_dir,direct,files),
                        }
            # If include_masks is False
            else:
                # Extracts text, gloss, and signer information
                for s in tmp:
                    seq_id = s["name"]
                    if seq_id in samples:
                        assert samples[seq_id]["name"] == s["name"]
                        assert samples[seq_id]["signer"] == s["signer"]
                        assert samples[seq_id]["gloss"] == s["gloss"]
                        assert samples[seq_id]["text"] == re.sub(r'[^\w\s]',' ',s["text"].replace("'",""))
                        samples[seq_id]["sign"] = torch.cat(
                            [samples[seq_id]["sign"], s["sign"]].to('cpu'), axis=1
                        )
                    else:
                        samples[seq_id] = {
                            "name": s["name"],
                            "signer": s["signer"],
                            "gloss": s["gloss"],
                            "text": re.sub(r'[^\w\s]',' ',s["text"].replace("'","")),
                            "sign": s["sign"].to('cpu'),
                        }
        # Converts dataset entries into torchtext.data.Example objects
        examples = []
        for s in samples:
            sample = samples[s]
            if include_masks:
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            sample["sign"],
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                            sample["sign"],
                        ],
                        fields,
                    )
                )
            else:
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            # This is for numerical stability
                            sample["sign"] + 1e-8,
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                        ],
                        fields,
                    )
                )
        # Passes processed examples and fields to torchtext's dataset class
        super().__init__(examples, fields, **kwargs)
