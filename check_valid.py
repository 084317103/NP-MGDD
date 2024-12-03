from rdkit import Chem
input_files = [
    "NP-MGDD/datasets/zuhe/data_before/npass.txt",
    "NP-MGDD/datasets/zuhe/data_before/tcmbank.txt",
    "NP-MGDD/datasets/zuhe/data_before/inflamnat.txt",
    "NP-MGDD/datasets/zuhe/data_before/cmaup.txt"
]

output_dir = "NP-MGDD/datasets/zuhe/data_remove_invalid/"
os.makedirs(output_dir, exist_ok=True)

for input_file in input_files:
    output_file = input_file.replace(".txt", "_after.txt")
    
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                outfile.write(smiles + "\n")
