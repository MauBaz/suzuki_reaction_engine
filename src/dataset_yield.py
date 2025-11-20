import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import numpy as np

ATOM_TYPES = ["H","B","C","N","O","F","Si","P","S","Cl","Br","I"]
ATOM_TYPE_TO_IDX = {a:i for i,a in enumerate(ATOM_TYPES)}

def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    return mol

def mol_to_graph(mol: Chem.Mol) -> Data:
    xs = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx = ATOM_TYPE_TO_IDX.get(sym, len(ATOM_TYPES))
        one_hot = [0]*(len(ATOM_TYPES)+1)
        one_hot[idx] = 1
        xs.append(one_hot)
    x = torch.tensor(xs, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i,j])
        edge_index.append([j,i])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

class SuzukiYieldDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sm1 = row["reactant_1_smiles"]
        sm2 = row["reactant_2_smiles"]
        mol_r = mol_from_smiles(sm1 + "." + sm2)
        graph = mol_to_graph(mol_r)
        y = torch.tensor([float(row["yield"])], dtype=torch.float)

        graph.y = y
        return graph
