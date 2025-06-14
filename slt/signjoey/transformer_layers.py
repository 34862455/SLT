# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        # input is split into multiple attention heads
        self.head_size = head_size = size // num_heads
        self.model_size = size # Total model size
        self.num_heads = num_heads

        # Linear projections for queries (Q), keys (K), and values (V)
        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        # # Linear layer to merge outputs from all heads
        self.output_layer = nn.Linear(size, size)
        # get attention weights
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.
        # Tensors of shape [Batch, Sequence Length, Embedding Dim].
        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        # masking tensor that prevents attention to padding tokens.
        :param mask: optional mask [B, 1, M]
        :return:
        """
        # For further processing
        batch_size = k.size(0)
        num_heads = self.num_heads

        # Passes k, q, and v through separate linear layers.
        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [Batch, Sequence, num_heads, head_size].
        # swaps dimensions to [Batch, num_heads, Sequence, head_size]
        # allows parallel processing of multiple attention heads.
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        # Prevents large gradients when head_size is large
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        # gives attention weights
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # Applies masking to prevent unwanted attention
        # Prevents decoder from looking at future tokens
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors
        # normalizes scores into probabilities.
        attention = self.softmax(scores)
        # helps prevent overfitting.
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        # Reshapes the output back to [Batch, Sequence, Embedding Dim]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        # Projects multi-head output back to model_size using self.output_layer
        output = self.output_layer(context)

        return output


# pylint: disable=arguments-differ
# feed-forward layer applied independently to each position in a sequence
# help model complex relationships after self-attention.
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        # Two linear transformations with ReLU activation in between
        super(PositionwiseFeedForward, self).__init__()
        # normalises activations
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        # transforms each token independently, refining features extracted by attention.
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    # Improve gradient flow
    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x #Residual Connection (+x) maintains input information.


# pylint: disable=arguments-differ
# injects positional information into embeddings
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    # use sin/cos to encode position
    #     Generalizes to unseen lengths (unlike learned embeddings).
    #     Encodes relative positions (important for attention).
    #     Smooth interpolation between positions.
    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        # Embedding dimension (must be even, since we alternate sin and cos).
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size) # Initialize with zeros
        position = torch.arange(0, max_len).unsqueeze(1) # column vector representing time steps
        # scales the sine/cosine wave frequencies
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        # Even indices get sine values.
        # Odd indices get cosine values.
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # Expands to batch-compatible format
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        # stores pe in model memory without being trainable
        self.register_buffer("pe", pe)
        self.dim = size

    # Takes input embeddings (emb).
    # Adds positional encodings (only up to sequence length).
    # Ensures word order information is retained in the Transformer.
    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


# Multiple layers of this class are stacked in the full Transformer encoder
class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        A single Transformer layer.
        :param size: (512 in standard Transformer)
        :param ff_size: (usually 4 * size)
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        # call
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        # call
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        # capture relationships between words.
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x #residual connection
        o = self.feed_forward(h)
        return o
#     Final Output is passed to the next Transformer layer


# Multiple layers of this class are stacked in the full Transformer decoder
class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality (512 in standard Transformers)
        :param ff_size: size of the feed-forward intermediate layer (usually 4 * size)
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        # call
        # Masked Self-Attention: Prevents future token peeking
        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        # Cross-Attention: Attends to encoder outputs
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        # call
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations (encoder outputs)
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        # Masked Self-Attention
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x #residual connection

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        # Cross-Attention
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o
