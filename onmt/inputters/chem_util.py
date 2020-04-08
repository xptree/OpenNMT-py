#!/usr/bin/env python
# encoding: utf-8
# File Name: chem_util.py
# Author: Jiezhong Qiu
# Create Time: 2020/04/07 18:22
# TODO:

import os.path as osp
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem
import pybel
import openbabel
import torch
import numpy as np
from collections import defaultdict
import dgl

def _obatom_coordinate(obatom):
    return np.array([obatom.x(), obatom.y(), obatom.z()])

def _obatom_feature(obmol):
    atom_feats_dict = defaultdict(list)
    for obatom in openbabel.OBMolAtomIter(obmol):
        atomic_num = obatom.GetAtomicNum()
        aromatic = obatom.IsAromatic()

        # The hybridization of this atom: 1 for sp, 2 for sp2, 3 for sp3, 4 for sq. planar, 5 for trig. bipy, 6 for octahedral
        hyb = obatom.GetHyb()

        implicit_num_h = obatom.ImplicitHydrogenCount()
        explicit_num_h = obatom.ExplicitHydrogenCount()
        acceptor = obatom.IsHbondAcceptor()
        donor = obatom.IsHbondDonor()

        atomic_mass = obatom.GetAtomicMass()
        # TODO is atom type useful?
        #  atomic_type = obatom.GetType()
        valence = obatom.GetValence()
        coordinate = [obatom.x(), obatom.y(), obatom.z()]

        atom_feats_dict['coord'].append(torch.FloatTensor(coordinate))
        atom_feats_dict['atomic_num'].append(atomic_num)
        atom_feats_dict['hyb'].append(hyb)

        h_u = [acceptor, donor, aromatic, implicit_num_h, explicit_num_h, atomic_mass, valence]
        atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

    atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['coord'] = torch.stack(atom_feats_dict['coord'], dim=0)
    atom_feats_dict['atomic_num'] = torch.LongTensor(atom_feats_dict['atomic_num'])
    atom_feats_dict['hyb'] = torch.LongTensor(atom_feats_dict['hyb'])
    return atom_feats_dict

def _obbond_feature(obmol, self_loop=False):
    bond_feats_dict = defaultdict(list)
    us, vs = [], []

    for obbond in openbabel.OBMolBondIter(obmol):
        begin = obbond.GetBeginAtomIdx()
        end = obbond.GetEndAtomIdx()
        assert begin > 0 and end > 0 # idx should be 1-based
        assert begin != end
        order = obbond.GetBondOrder()
        assert order > 0

        # add edge begin->end
        bond_feats_dict['order'].append(order)
        us.append(begin-1) # dgl node id is 0-based
        vs.append(end-1)

    bond_feats_dict['order'] = torch.LongTensor(bond_feats_dict['order'])

    return us, vs, bond_feats_dict

def openbabel_to_dgl_graph(smi):
    mol = pybel.readstring("smi", smi)
    mol.addh() # add hydrogens, if this function is not called, pybel will output xyz string with no hydrogens.
    mol.make3D(forcefield='mmff94', steps=100)
    # possible forcefields: ['uff', 'mmff94', 'ghemical']
    mol.localopt()
    obmol = mol.OBMol
    # add nodes
    num_atoms = obmol.NumAtoms()
    g = dgl.DGLGraph()
    atom_feats = _obatom_feature(obmol)
    g.add_nodes(num=num_atoms, data=atom_feats)

    us, vs, bond_feats = _obbond_feature(obmol)
    g.add_edges(us, vs, bond_feats)
    g.readonly()
    return g

def alchemy_nodes(mol):
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

def alchemy_edges(mol, self_loop=True):
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

def rdkit_to_dgl_graph(smi, self_loop=True):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    g = dgl.DGLGraph()

    # add nodes
    num_atoms = mol.GetNumAtoms()
    atom_feats = alchemy_nodes(mol)
    g.add_nodes(num=num_atoms, data=atom_feats)
    g.readonly()
    return g
    if self_loop:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms)],
            [j for i in range(num_atoms) for j in range(num_atoms)])
    else:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                j for i in range(num_atoms)
                for j in range(num_atoms) if i != j
            ])

    bond_feats = alchemy_edges(mol, self_loop)
    g.edata.update(bond_feats)

