def extract_smiles(file_path):
    smiles_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    smiles_set.add(smiles)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    smiles_set.add(smiles)
    return smiles_set

file_paths = ['NPMGDD/datasets/zuhe/tcmbank_after.txt', 'NPMGDD/datasets/zuhe/npass_after.txt', 'NPMGDD/datasets/zuhe/inflamnat.txt', 'NPMGDD/datasets/zuhe/cmaup.txt']

unique_smiles = set()
for file_path in file_paths:
    unique_smiles.update(extract_smiles(file_path))

with open('NPMGDD/datasets/zuhe/merged_smiles.txt', 'w') as f:
    for smiles in unique_smiles:
        f.write(smiles + '\n')
