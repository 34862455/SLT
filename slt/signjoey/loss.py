# coding: utf-8
"""
Module to implement training loss
"""

# Implements Cross-Entropy Loss (XentLoss).
# Supports Label Smoothing (to improve generalization).
# Uses Negative Log-Likelihood (NLL) Loss or Kullback-Leibler (KL) Divergence depending on smoothing.

import torch
from torch import nn, Tensor
# Stores target distributions (with gradient tracking)
from torch.autograd import Variable


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        # Label smoothing factor
        self.smoothing = smoothing
        # index of the padding token (ignored during loss computation)
        self.pad_index = pad_index
        # Standard Cross-Entropy Loss
        if self.smoothing <= 0.0:
            # Uses Negative Log-Likelihood Loss (NLLLoss)
            # Ignores padding tokens (ignore_index=self.pad_index
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # Uses Kullback-Leibler Divergence Loss (KLDivLoss)
            # Helps prevent the model from becoming overconfident
            self.criterion = nn.KLDivLoss(reduction="sum")

    # Applying Label Smoothing
    # Reduces overfitting by preventing the model from assigning full probability (1.0) to the correct class
    # Instead of one-hot encoding, we distribute some probability mass across all tokens
    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # Creates an empty probability distribution
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        # Distributes probability mass uniformly to non-padding tokens
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        # Converts smooth_dist into a non-trainable variable
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    # Compute Loss
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            # Ensures dimensions match between log_probs and targets
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        # Computes final loss using either NLLLoss or KLDivLoss
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss

