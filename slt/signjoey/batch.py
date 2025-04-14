# coding: utf-8
import math
import random
import torch
import numpy as np

# extends torchtext batch objects by adding:
#     Sign features (sgn).
#     Gloss sequences (gls).
#     Text sequences (txt).
#     Masks for padding and attention.
#     Support for frame subsampling & masking.

# holds a single batch of sign language data
class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        is_train: bool = False,
        # Moves batch to GPU if use_cuda=True
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
        include_masks: bool = False,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        # Stores metadata (sequence and signer IDs
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        # Extracts sign language video features (sgn) and sequence lengths (sgn_lengths)
        self.sgn, self.sgn_lengths = torch_batch.sgn

        # Here be dragons
        # Frame Subsampling
        # Data augmentation?
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                features = features.clone()
                # improve generalization during training
                # randomly selects an offset for subsampling
                if random_frame_subsampling and is_train:
                    # Randomly subsamples frames to reduce input size
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0 : tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        # Frame Masking
        # Data augmentation
        if random_frame_masking_ratio and is_train:
            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = (
                (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            )
            for idx, features in enumerate(self.sgn):
                features = features.clone()
                # Randomly masks some frames in sign language video features
                # Helps simulate missing frames (similar to dropout)
                mask_frame_idx = np.random.permutation(
                    int(self.sgn_lengths[idx].long().numpy())
                )[: num_mask_frames[idx]]
                # Frames are replaced with near-zero values
                features[mask_frame_idx, :] = 1e-8
                tmp_sgn[idx] = features
            self.sgn = tmp_sgn

        # Computes a mask for padding in sgn (sign features)
        self.sgn_dim = sgn_dim
        if include_masks:
            # uses precomputed masks
            sgn_mask = torch_batch.mask
            self.sgn_lengths = (torch.sum(sgn_mask,dim=1)).float()
            self.sgn_mask = (sgn_mask == 1).unsqueeze(1)
            
        else:
            # creates a mask from zero-padding
            self.sgn_mask = (self.sgn != 0).any(dim=2, keepdim=True).transpose(1, 2)
        

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        # Extracts text sequences and lengths
        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:] # Shifted by one
            # we exclude the padded areas from the loss computation
            # Creates a mask to ignore padding.
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        # Extracts gloss sequences and lengths
        if hasattr(torch_batch, "gls"):
            self.gls, self.gls_lengths = torch_batch.gls
            # Counts the total number of gloss tokens
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()
        
        # Moves all tensors to GPU for faster training
        if use_cuda:
            # call
            self._make_cuda()

    # Moves sign, text, and mask tensors to GPU
    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

    # Sorts the batch by sequence length
    # Mostly used for efficient processing for RNNs
    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
