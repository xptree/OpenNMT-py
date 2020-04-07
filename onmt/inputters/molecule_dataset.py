#!/usr/bin/env python
# encoding: utf-8
# File Name: molecule_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2020/04/05 20:09
# TODO:

import six
import os.path as osp
from collections import defaultdict
from torchtext.data import RawField

import torch
from onmt.inputters.datareader_base import DataReaderBase
from onmt.inputters.text_dataset import text_fields

# domain specific dependencies
try:
    import rdkit
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem import AllChem
    import dgl
    import numpy as np
except ImportError:
    rdkit, dgl, np = None, None, None

class MoleculeDataReader(DataReaderBase):
    """Read molecular graph data from dist.

    Args:

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing any of ``rdkit`` or ``dgl`` fail.
    """

    def __init__(self, self_loop=True):
        self._check_deps()
        self.self_loop = self_loop

    @classmethod
    def from_opt(cls, opt):
        return cls(self_loop=True)

    @classmethod
    def _check_deps(cls):
        if any([rdkit is None, dgl is None, np is None]):
            cls._raise_missing_dep(
                    "rdkit", "dgl", "numpy")

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

class MolecularGraphField(RawField):
    """Defines a molecular graph datatype and instructions for converting to DGL graph.

    See :class:`RawField` for attribute descriptions.
    """


    def __init__(self, self_loop=True):
        super(MolecularGraphField, self).__init__()
        self.self_loop = self_loop

    def process(self, batch, *args, **kwargs):
        """Convert outputs of preprocess into DGLgraphs.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        batch_graphs = dgl.batch(batch)
        return batch_graphs

    def alchemy_nodes(self, mol):
        """Featurization for all atoms in a molecule. The atom indices
        will be preserved.

        Args:
            mol : rdkit.Chem.rdchem.Mol
              RDKit molecule object
        Returns
            atom_feats_dict : dict
              Dictionary for atom features
        """
        atom_feats_dict = defaultdict(list)
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
            atom_feats_dict['node_type'].append(atom_type)

            h_u = []
            h_u += [
                int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
            ]
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)
            ]
            h_u.append(num_h)
            atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'],
                                                dim=0)
        atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(
            atom_feats_dict['node_type'])

        return atom_feats_dict

    def alchemy_edges(self, mol, self_loop=True):
        """Featurization for all bonds in a molecule. The bond indices
        will be preserved.

        Args:
          mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
          bond_feats_dict : dict
              Dictionary for bond features
        """
        bond_feats_dict = defaultdict(list)

        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self_loop:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict['e_feat'].append([
                    float(bond_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE,
                              Chem.rdchem.BondType.TRIPLE,
                              Chem.rdchem.BondType.AROMATIC, None)
                ])
                bond_feats_dict['distance'].append(
                    np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict['e_feat'] = torch.FloatTensor(
            bond_feats_dict['e_feat'])
        bond_feats_dict['distance'] = torch.FloatTensor(
            bond_feats_dict['distance']).reshape(-1, 1)

        return bond_feats_dict

    def preprocess(self, x):
        # <RC_1> C S ( = O ) ( = O ) c 1 c c c ( O c 2 c c ( Cl ) c c c 2 C C C ( = O ) O ) c ( C ( F ) ( F ) F ) c 1
        x = x.split(" ")
        if x[0].startswith("<RC_"):
            x = x[1:]
        x = "".join(x)
        mol = Chem.MolFromSmiles(x)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        g = dgl.DGLGraph()

        # add nodes
        num_atoms = mol.GetNumAtoms()
        atom_feats = self.alchemy_nodes(mol)
        g.add_nodes(num=num_atoms, data=atom_feats)
        return g
        if self.self_loop:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms)],
                [j for i in range(num_atoms) for j in range(num_atoms)])
        else:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                    j for i in range(num_atoms)
                    for j in range(num_atoms) if i != j
                ])

        bond_feats = self.alchemy_edges(mol, self.self_loop)
        g.edata.update(bond_feats)

        return g

class MoleculeField(RawField):
    def __init__(self, graph_field, text_field):
        super(MoleculeField, self).__init__()
        self.graph_field = graph_field
        self.text_field = text_field

    def process(self, batch, device=None):
        batch_graph, batch_text = batch

        data_graph = self.graph_field.process(batch_graph, device)
        data_text = self.text_field.process(batch_text, device)

        if self.text_field.base_field.include_lengths:
            return data_graph, data_text[0], data_text[1]
        else:
            return data_graph, data_text

    def preprocess(self, x):
        return self.graph_field.preprocess(x), self.text_field.preprocess(x)

def molecule_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[1]), len(ex.tgt[0])
    return len(ex.src[1])

def molecule_fields(**kwargs):
    nesting_text_field = text_fields(**kwargs)
    nesting_graph_field = MolecularGraphField(self_loop=True)
    mol_field = MoleculeField(nesting_graph_field, nesting_text_field)
    return mol_field
