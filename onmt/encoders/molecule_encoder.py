#!/usr/bin/env python
# encoding: utf-8
# File Name: molecule_encoder.py
# Author: Jiezhong Qiu
# Create Time: 2020/04/08 17:31
# TODO:

from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder

class MoleculeEncoder(EncoderBase):
    """A GNN-Transformer two-way encoder for molecule src.

    Args:
        num_layers (int): number of transformer encoder layers
        d_model (int): size of the transformer model
        heads (int): number of transformer heads
        d_ff (int): size of the transformer inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """
    def __init__(self, d_model, graph_encoder, sequence_encoder):
        super(MoleculeEncoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.sequence_encoder = sequence_encoder

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        graph_encoder = None
        sequence_encoder = TransformerEncoder.from_opt(opt, embeddings)
        return cls(
            opt.enc_rnn_size,
            graph_encoder,
            sequence_encoder
            )

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src.text, lengths)
        return self.sequence_encoder(src.text, lengths)

    def update_dropout(self, dropout, attention_dropout):
        self.sequence_encoder.update_dropout(dropout, attention_dropout)
