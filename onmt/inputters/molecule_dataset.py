#!/usr/bin/env python
# encoding: utf-8
# File Name: molecule_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2020/04/05 20:09
# TODO:

from functools import partial
import six
import torch
import json
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase
from onmt.inputters.text_dataset import _feature_tokenize, TextMultiField
from onmt.utils.logging import logger
from .chem_util import openbabel_to_dgl_graph
from joblib import Parallel, delayed

# domain specific dependencies
try:
    import rdkit
    import openbabel
    import pybel
    import dgl
    import dgl.data
    import numpy as np
except ImportError:
    rdkit, openbabel, pybel, dgl, np = None, None, None, None, None


class MoleculeBatcher(object):
    def __init__(self, graph=None, text=None):
        self.graph = graph
        self.text = text
    def to(self, device):
        device = torch.device(device)
        self.graph = self.graph.to(device)
        self.text = self.text.to(device)
        return self

class MoleculeDataReader(DataReaderBase):
    """Read molecular graph data from dist.

    Args:

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing any of ``rdkit``, ``openbabel``, ``pybel``, ``dgl`` or ``numpy`` fail.
    """

    def __init__(self, self_loop=True):
        self._check_deps()
        self.self_loop = self_loop

    @classmethod
    def from_opt(cls, opt):
        return cls(self_loop=True)

    @classmethod
    def _check_deps(cls):
        if any([rdkit is None, openbabel is None, pybel is None, dgl is None, np is None]):
            cls._raise_missing_dep(
                    "rdkit", "openbabel", "pybel", "dgl", "numpy")

    def read(self, molecules, side, _dir=None):
        """Read data into dicts.

        Args:
            molecules (str or Iterable[str]):
                path to molecules file or iterable of the actual molecule data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(molecules, str):
            molecules = DataReaderBase._read_file(molecules)
        for i, mol in enumerate(molecules):
            if isinstance(mol, six.binary_type):
                mol = mol.decode("utf-8")
            yield {side: mol, "indices": i}

def _to_dgl_graph(index, smi):
    g = openbabel_to_dgl_graph(smi)
    #  g = rdkit_to_dgl_graph(x)
    g.readonly()
    return index, smi, g


class MoleculeStorage(object):
    def __init__(self, memo=None):
        self.memo = memo if memo else {}
        logger.info("initialize storage with %d molecules" % len(self.memo))

    @classmethod
    def from_file(cls, graph_dir="/data/jiezhong/retrosynthesis/transformer_c2c_mit_database_new_all_data/data/USPTO-50K/mol.bin", smi_dir="/data/jiezhong/retrosynthesis/transformer_c2c_mit_database_new_all_data/data/USPTO-50K/mol2.json"):
        logger.info("going to load molecular graphs from %s" % graph_dir)
        g_list, _ = dgl.data.load_graphs(graph_dir)
        logger.info("going to load SMILES from %s" % smi_dir)
        with open(smi_dir, "r") as f:
            smi_list = json.load(f)
        memo = dict(zip(smi_list, g_list))
        return cls(memo)

    def query(self, batch):
        # <RC_1> C S ( = O ) ( = O ) c 1 c c c ( O c 2 c c ( Cl ) c c c 2 C C C ( = O ) O ) c ( C ( F ) ( F ) F ) c 1
        to_compute, cached = [], [None] * len(batch)
        for i, x in enumerate(batch):
            x = "".join(x[1:])
            if x in self.memo:
                cached[i] = self.memo[x]
            else:
                to_compute.append((i, x))
        if len(to_compute):
            computed = Parallel(n_jobs=32)(delayed(_to_dgl_graph)(i, x) for i, x in to_compute)
            for i, smi, graph in computed:
                self.memo[smi] = graph
                cached[i] = graph
        return cached


class MoleculeField(TextMultiField):
    def __init__(self, base_name, base_field, feats_fields):
        super(MoleculeField, self).__init__(
                base_name=base_name,
                base_field=base_field,
                feats_fields=feats_fields)

    def process(self, batch, device=None):
        batch_by_feat = list(zip(*batch))
        #  graph_data = Parallel(n_jobs=48)(delayed(_to_dgl_graph)(x) for x in batch_by_feat[0])

        if not hasattr(self, 'storage'):
            self.storage = MoleculeStorage.from_file()

        graph_data = self.storage.query(batch_by_feat[0])
        graph_data = dgl.batch(graph_data)
        if device:
            graph_data = graph_data.to(torch.device(device))

        text_data = super().process(batch, device)
        if self.base_field.include_lengths:
            data = MoleculeBatcher(graph=graph_data, text=text_data[0])
            return data, text_data[1]
        else:
            data = MoleculeBatcher(graph=graph_data, text=text_data)
            return data

def molecule_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    mol_field = MoleculeField(fields_[0][0], fields_[0][1], fields_[1:])
    return mol_field
