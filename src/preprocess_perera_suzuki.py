import pandas as pd
from rdkit import Chem
import os

infile = "data/raw/aap9112_Data_File_S1.xlsx"
outfile = "data/processed/suzuki_products.csv"

df = pd.read_excel(infile)

# Rename columns based on your screenshot
rename_map = {
    "Reactant_1_Name": "aryl_halide_name",
    "Reactant_2_Name": "boronic_partner_name",
    "Ligand_Short_Hand": "ligand",
    "Ligand_eq": "ligand_eq",
    "Catalyst_1_Short_Hand": "base",  # Catalyst is the base
    "Catalyst_1_eq": "base_eq",
    "Solvent_1_Short_Hand": "solvent",
    "Reagent_1_Short_Hand": "reagent",  
    "Product_Yield_PCT_Area_UV": "yield_pct"
}

df = df.rename(columns={k: v for k,v in rename_map.items() if k in df.columns})

# Complete SMILES mappings from RXN yields website
aryl_smiles = {
    '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12', 
    '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12', 
    '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O',
    '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12', 
    '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl',
    'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+]',
    '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3'
}

boronic_smiles = {
    '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O', 
    '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4', 
    '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br' 
}

# Ligand SMILES from RXN yields
ligand_smiles = {
    'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'PCy3': 'C1CCCCC1P(C1CCCCC1)C1CCCCC1',
    'P(Ph)3': 'c3c(P(c1ccccc1)c2ccccc2)cccc3',
    'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C',
    'CataCXium A': 'CC(C)C1=CC(=C(C(=C1)C(C)C)P(C2CCCCC2)C3CCCCC3)C(C)C',
    'P(o-Tol)3': 'CC1=C(C=CC=C1)P(C2=C(C=CC=C2)C)C3=C(C=CC=C3)C',
    'XPhos': 'CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(=CC=C1)C(C)C)C',
    'SPhos': 'COC1=CC=CC(=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)OC',
    'P(fur)3': 'O1C=CC=C1P(C2=CC=CO2)C3=CC=CO3',
    'dppf': 'C1=CC=C(C=C1)P([Fe]CCCCP(C2=CC=CC=C2)C3=CC=CC=C3)C4=CC=CC=C4',
    'dppp': 'C1=CC=C(C=C1)P(CCCP(C2=CC=CC=C2)C3=CC=CC=C3)C4=CC=CC=C4',
    'Xantphos': 'CC1(C)C2=CC=CC=C2OC3=C1C=CC=C3P(C4=CC=CC=C4)C5=CC=CC=C5',
    'P(nBu)3': 'CCCCCCCCCCCCCCCCCCCCCCCCCC',
    'BINAP': 'C1=CC=C2C(=C1)C(=CC=C2)P(C3=CC=CC=C3)C4=C(C5=CC=CC=C5C=C4)P(C6=CC=CC=C6)C7=CC=CC=C7',
    'dtbpf': 'CC(C)(C)P([Fe]CCCCP(C(C)(C)C)C(C)(C)C)C(C)(C)C'
}

# Base/Catalyst SMILES
base_smiles = {
    'Cs2CO3': '[Cs+].[Cs+].[O-]C([O-])=O',
    'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O',
    'Na2CO3': '[Na+].[Na+].[O-]C([O-])=O',
    'CsF': '[F-].[Cs+]',
    'K2CO3': '[K+].[K+].[O-]C([O-])=O',
    'KF': '[F-].[K+]',
    'Et3N': 'CCN(CC)CC',
    'BTMG': 'CN(C)C(=NC1CCCCC1)NC2CCCCC2',
    'KOH': '[OH-].[K+]',
    'NaOtBu': 'CC(C)(C)[O-].[Na+]',
    'NaOH': '[OH-].[Na+]',
    'KOtBu': 'CC(C)(C)[O-].[K+]'
}

# Solvent SMILES
solvent_smiles = {
    'MeCN': 'CC#N',
    'THF': 'C1CCOC1',
    'DMA': 'CN(C)C=O',
    'Tol': 'CC1=CC=CC=C1',
    'Dioxane': 'C1COCCO1',
    'DMF': 'CN(C)C=O',
    'MeOH': 'CO',
    'EtOH': 'CCO',
    'BuOH': 'CCCCO',
    'Water': 'O',
    'DMSO': 'CS(C)=O',
    'NMP': 'CN1CCCC1=O'
}

# Map to SMILES
df["reactant_1_smiles"] = df["aryl_halide_name"].map(aryl_smiles)
df["reactant_2_smiles"] = df["boronic_partner_name"].map(boronic_smiles)
df["ligand_smiles"] = df["ligand"].map(ligand_smiles)
df["base_smiles"] = df["base"].map(base_smiles)
df["solvent_smiles"] = df["solvent"].map(solvent_smiles)

# Keep equivalents info
df["ligand_eq"] = pd.to_numeric(df["ligand_eq"], errors='coerce').fillna(0)
df["base_eq"] = pd.to_numeric(df["base_eq"], errors='coerce').fillna(0)

# Drop rows with missing critical SMILES
df = df.dropna(subset=["reactant_1_smiles", "reactant_2_smiles"])

# Output with all relevant columns
out_df = df[[
    "reactant_1_smiles", "reactant_2_smiles", 
    "ligand_smiles", "ligand_eq",
    "base_smiles", "base_eq",
    "solvent_smiles",
    "yield_pct"
]]
out_df = out_df.rename(columns={"yield_pct": "yield"})

os.makedirs(os.path.dirname(outfile), exist_ok=True)
out_df.to_csv(outfile, index=False)
print(f"Wrote {len(out_df)} records to {outfile}")