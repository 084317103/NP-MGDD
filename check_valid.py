from rdkit import Chem
input_files = [
    "NPMGDD/datasets/zuhe/npass.txt",
    "NPMGDD/datasets/zuhe/tcmbank.txt",
    "NPMGDD/datasets/zuhe/inflamnat.txt",
    "NPMGDD/datasets/zuhe/cmaup.txt"
]
for input_file in input_files:
    output_file = input_file.replace(".txt", "_after.txt")
    
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                outfile.write(smiles + "\n")
