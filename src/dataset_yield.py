import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np

ATOM_TYPES = ["H","B","C","N","O","F","P","S","Cl","Br","I","K","Na","Cs","Fe"]
ATOM_TYPE_TO_IDX = {a:i for i,a in enumerate(ATOM_TYPES)}

def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def advanced_atom_features(atom):
    features = []
    atom_type = atom.GetSymbol()
    atom_idx = ATOM_TYPE_TO_IDX.get(atom_type, len(ATOM_TYPES))
    one_hot = [0] * (len(ATOM_TYPES) + 1)
    one_hot[atom_idx] = 1
    features.extend(one_hot)
    
    features.extend([
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 2,
        int(atom.GetHybridization()) / 4,
        int(atom.GetIsAromatic()),
        atom.GetMass() / 100,
        int(atom.IsInRing()),
    ])
    
    return features

def get_molecular_features(smiles):
    """Extract molecular descriptors for conditions"""
    if pd.isna(smiles) or smiles is None:
        return np.zeros(10)
    
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(10)
    
    features = [
        Descriptors.MolWt(mol) / 500,
        Descriptors.MolLogP(mol) / 5,
        Descriptors.NumHDonors(mol) / 10,
        Descriptors.NumHAcceptors(mol) / 10,
        Descriptors.TPSA(mol) / 140,
        Descriptors.NumRotatableBonds(mol) / 10,
        rdMolDescriptors.CalcNumRings(mol) / 5,
        rdMolDescriptors.CalcNumAromaticRings(mol) / 5,
        rdMolDescriptors.CalcNumHeteroatoms(mol) / 10,
        Descriptors.FractionCSP3(mol)
    ]
    return np.array(features)

def mol_to_graph(mol: Chem.Mol) -> Data:
    xs = []
    for atom in mol.GetAtoms():
        xs.append(advanced_atom_features(atom))
    x = torch.tensor(xs, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_index.extend([[i,j], [j,i]])
        
        bond_features = [
            int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        edge_attr.extend([bond_features, bond_features])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,6), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class SuzukiYieldDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        
        # Normalize yields
        self.df['yield'] = self.df['yield'] / 100.0 if self.df['yield'].max() > 1.5 else self.df['yield']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Reactants graph
        sm1 = row["reactant_1_smiles"]
        sm2 = row["reactant_2_smiles"]
        mol_r = mol_from_smiles(sm1 + "." + sm2)
        
        if mol_r is None:
            return None
            
        graph = mol_to_graph(mol_r)
        
        # Extract condition features
        ligand_feat = get_molecular_features(row.get("ligand_smiles", None))
        base_feat = get_molecular_features(row.get("base_smiles", None))
        solvent_feat = get_molecular_features(row.get("solvent_smiles", None))
        
        # Equivalents (normalized)
        ligand_eq = float(row.get("ligand_eq", 0)) / 10.0
        base_eq = float(row.get("base_eq", 0)) / 10.0
        
        # Concatenate all condition features
        # Concatenate all condition features
        conditions = torch.tensor(
        np.concatenate([ligand_feat, base_feat, solvent_feat, [ligand_eq, base_eq]]),
        dtype=torch.float
        ).unsqueeze(0)  # Make it 2D: [1, 32] for proper batching
        
        graph.conditions = conditions
        graph.y = torch.tensor([float(row["yield"])], dtype=torch.float)
        
        return graph