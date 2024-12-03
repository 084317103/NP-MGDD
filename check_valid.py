from rdkit import Chem
input_file = "autodl-tmp/MolGPT/datasets/zuhe/cmaup.txt"
output_file = "autodl-tmp/MolGPT/datasets/zuhe/cmaup_after.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        smiles = line.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            outfile.write(smiles + "\n")