# coding:latin-1
import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from config import get_parser
from model import CharRNN
from trainer import CharRNNTrainer

import math
import re
import os


import argparse
import os
import sys
import torch
import rdkit

from moses.script_utils import read_smiles_csv


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']

def get_dataset(split='train'):
    """
    Loads MOSES dataset from a text file.

    Arguments:
        split (str): Split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    
    # Construct the new path to the text file
    base_path = os.path.dirname(__file__)
    path = os.path.join('/root/autodl-tmp/charRNN/datasets/', split + '_smiles.txt')
    
    # Read SMILES strings from the text file
    with open(path, 'r') as file:
        smiles = file.read().splitlines()
    smiles = np.array(smiles)
    return smiles




def main(config):
    set_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly

    if config.train_load is None:
        train_data = get_dataset('train')
    else:
        train_data = read_smiles_csv(config.train_load)
    if config.val_load is None:
        val_data = get_dataset('test')
    else:
        val_data = read_smiles_csv(config.val_load)
     

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), \
            'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    
   

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)
        
    model = CharRNN(vocab, config)
    model_state = torch.load(config.model_load)
    model.load_state_dict(model_state)
    trainer = CharRNNTrainer(config, model)
    trainer.fit_ahc(model, train_data, val_data)
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    main(config)

   
    
