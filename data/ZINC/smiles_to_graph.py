import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA


""" To replicate the featurizer from the paper, code was taken from: 
https://github.com/SeongokRyu/augmented-GCN/blob/master/database/smilesToGraph.py
https://github.com/SeongokRyu/augmented-GCN/blob/master/database/calcProperty.py
"""

def adj_k(adj, k):

    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)  

    return convertAdj(ret)

def convertAdj(adj):

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d

def convertToGraph_list(smiles_list, k):
    adj = []
    features = []
    for i in smiles_list:
        # Mol
        iadj, feature = convertToGraph(i, k)
        adj.append(iadj)
        features.append(feature)
    features = np.asarray(features)
    adj = np.array(adj)
    return adj, features

def convertToGraph(smile, k):
    maxNumAtoms = 50
    iMol = Chem.MolFromSmiles(smile.strip())
    #Adj
    iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
    # Feature
    if( iAdjTmp.shape[1] <= maxNumAtoms):
        # Feature-preprocessing
        iFeature = np.zeros((maxNumAtoms, 58))
        iFeatureTmp = []
        for atom in iMol.GetAtoms():
            iFeatureTmp.append( atom_feature(atom) ) ### atom features only
        iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp ### 0 padding for feature-set
        feature = iFeature
        # Adj-preprocessing
        iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
        iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
        adj = adj_k(np.asarray(iAdj), k)
    return adj, feature
    

def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_prop(smile, property):
    prop_map = {"TPSA": CalcTPSA, "LOGP": MolLogP, "QED": qed}
    smile = smile.strip()
    m = Chem.MolFromSmiles(smile)
    return prop_map[property](m)

