# NP-MGDD
  Our project's goal is to use the NP-MGDD method, which combines GPT and AHC, to generate drug-like molecules for drug discovery.
# 1. data collect 
We collected SMILES strings from four types of natural product libraries in the datasets/zuhe/data_before folder. Then, we removed the invalid SMILES in the datasets/zuhe/data_remove_invalid folder. After that, we merged these four natural product libraries, removing duplicate SMILES, resulting in the datasets/zuhe/merged_smiles.txt file.
# 2. data preprocess
Use data-preprocess.py to remove SMILES strings with a number of non-hydrogen atoms less than 10 (small molecules) or greater than 100 (large molecules)，and the processed SMILES strings are saved in the datasets/zuhe/merged_smiles_after.txt file.
# 3.pre-train
Use the script to perform the model pre-training process, where run_name is the name used to save the model.
~~~
python NP-MGDD/train/train.py --run_name pretrain --batch_size 128 --max_epochs 10
~~~
# 4.reinforcement learning
use the script to perform the reinforcement learning process，where run_name is the name used to save the model and model_weight indicates which model is used for reinforcement learning training.
~~~
python NP-MGDD/train/ahc.py --run_name ahc-300 --batch_size 64 --max_epochs 300 --model_weight autodl-tmp/MolGPT/cond_gpt/weights/pretrain.pt
~~~
# 5.generating
use the script to preform the generating process, where model_weight refers to the model used for generation, csv_name specifies the filename for saving the generated SMILES, and gen_size indicates the total number of generated molecules.
~~~
python autodl-tmp/MolGPT/generate/generate.py --model_weight autodl-tmp/MolGPT/cond_gpt/weights/ahc-300.pt --csv_name ahc_300_generation --gen_size 10000 --batch_size 128
~~~
