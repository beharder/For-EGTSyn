import csv
from itertools import islice
import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from util import *


def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1:]


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def bond_features(bond):
    return np.array(one_of_k_encoding_unk(bond.GetBondType(),[Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,'Unknown'])+
                    one_of_k_encoding(bond.GetBondDir(), [Chem.rdchem.BondDir.ENDUPRIGHT,
                    Chem.rdchem.BondDir.ENDDOWNRIGHT, Chem.rdchem.BondDir.NONE])
                    + [bond.GetIsAromatic()]+[bond.GetIsConjugated()]+[bond.IsInRing()]+[bond.HasOwningMol()]
                    +[0]*66)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def smile_to_gragh_edge(smile):
    mol = Chem.MolFromSmiles(smile)
    a_size = mol.GetNumAtoms()
    features = []
    edges = []
    for atom in mol.GetAtoms():
        atom_feature = atom_features(atom)
        features.append(atom_feature / sum(atom_feature))
    for bond in mol.GetBonds():
        bond_feature = bond_features(bond)
        features.append(bond_feature / sum(bond_feature))
        edges.append([bond.GetBeginAtomIdx(), bond.GetIdx() + a_size])
        edges.append([bond.GetIdx() + a_size, bond.GetEndAtomIdx()])
    g_edge = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g_edge.edges():
        edge_index.append([e1, e2])

    return a_size + mol.GetNumBonds(), features, edge_index


def creat_data(datafile, cellfile):
    file2 = cellfile
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)

    compound_iso_smiles = []
    df = pd.read_csv('data\smiles.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    smile_graph_atomedge = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        g_edge = smile_to_gragh_edge(smile)
        smile_graph[smile] = g
        smile_graph_atomedge[smile] = g_edge

    datasets = datafile
    # convert to PyTorch data format
    processed_data_file_train = 'data/new' + datasets + '_train.pt'

    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('data/' + datasets + '.csv')
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)

        print('creating graphs...')
        Create_data(root='data/', dataset=datafile + '_drug1', xd=drug1, xt=cell, xt_featrue=cell_features, y=label,
                       smile_graph=smile_graph)
        Create_data(root='data/', dataset=datafile + '_drug1_edge', xd=drug1, xt=cell, xt_featrue=cell_features, y=label,
                       smile_graph=smile_graph_atomedge)
        Create_data(root='data/', dataset=datafile + '_drug2', xd=drug2, xt=cell, xt_featrue=cell_features, y=label,
                       smile_graph=smile_graph)
        Create_data(root='data/', dataset=datafile + '_drug2_edge', xd=drug2, xt=cell, xt_featrue=cell_features, y=label,
                       smile_graph=smile_graph_atomedge)
        print('creating completed!')
        print('preparing ', datasets + '_.pt in pytorch format!')


if __name__ == "__main__":
    cell_data = 'data\cell_line_features_954.csv'
    com_data = ['combination_for_train']
    creat_data(com_data, cell_data)