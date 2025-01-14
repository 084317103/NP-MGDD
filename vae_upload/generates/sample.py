import argparse
import sys
import torch
import rdkit
import pandas as pd
from tqdm.auto import tqdm
from moses.script_utils import set_seed
from model import VAE
import re
import torch
import argparse


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def add_common_arg(parser):
    def torch_device(arg):
        if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
            raise argparse.ArgumentTypeError(
                'Wrong device format: {}'.format(arg)
            )

        if arg != 'cpu':
            splited_device = arg.split(':')

            if (not torch.cuda.is_available()) or \
                    (len(splited_device) > 1 and
                     int(splited_device[1]) > torch.cuda.device_count()):
                raise argparse.ArgumentTypeError(
                    'Wrong device: {} is not available'.format(arg)
                )

        return arg

    # Base
    parser.add_argument('--device',
                        type=torch_device, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')

    return parser

    
def get_parser():
    parser = argparse.ArgumentParser()
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")
    return parser


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    model = VAE(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    samples = []
    n = config.n_samples
    with tqdm(total=config.n_samples, desc='Generating samples') as T:
        while n > 0:
            current_samples = model.sample(
                min(n, config.n_batch), config.max_len
            )
            samples.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

    samples = pd.DataFrame(samples, columns=['SMILES'])
    samples.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    main(config)
