

### `src/preprocess_perera_suzuki.py`  

import pandas as pd
from rdkit import Chem
import os

infile = "data/raw/aap9112_Data_File_S1.xlsx"
outfile = "data/processed/suzuki_products.csv"

df = pd.read_excel(infile)

# Rename columns (adjust as per the actual Excel)
rename_map = {
    "Reactant_1_Name": "aryl_halide_name",
    "Reactant_2_Name": "boronic_partner_name",
    "Ligand_Short_Hand": "ligand",
    "Base_Short_Hand": "base",
    "Solvent_1_Short_Hand": "solvent",
    "Temperature_C": "temperature_C",
    "Product_Yield_PCT_Area_UV": "yield_pct"
}
df = df.rename(columns={k: v for k,v in rename_map.items() if k in df.columns})

# Drop missing reactants or yield
df = df.dropna(subset=["aryl_halide_name", "boronic_partner_name", "yield_pct"])

# Map names to SMILES — you must fill in full mapping based on supplementary info
aryl_smiles = {
    '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
    '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
    'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
    '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O'
}

boronic_smiles = {
    '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O', 
    '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4', 
    '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br' 
}

df["reactant_1_smiles"] = df["aryl_halide_name"].map(aryl_smiles)
df["reactant_2_smiles"] = df["boronic_partner_name"].map(boronic_smiles)

# Drop rows missing SMILES
df = df.dropna(subset=["reactant_1_smiles","reactant_2_smiles"])

# Build output CSV selecting reactants + yield only
out_df = df[["reactant_1_smiles","reactant_2_smiles","yield_pct"]]
out_df = out_df.rename(columns={"yield_pct":"yield"})

os.makedirs(os.path.dirname(outfile), exist_ok=True)
out_df.to_csv(outfile, index=False)
print(f"Wrote {len(out_df)} records to {outfile}")
