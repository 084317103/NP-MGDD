import argparse
import re
import torch


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
def add_train_args(parser):

    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load', type=str, help='Input data in csv format to train')
    common_arg.add_argument('--val_load', type=str, help="Input data in csv format for validation")
    common_arg.add_argument('--model_save', type=str, required=True, default='model.pt', help='Where to save the model')
    common_arg.add_argument('--save_frequency', type=int, default=6, help='How often to save the model')
    common_arg.add_argument('--log_file', type=str, required=False, help='Where to save the log')
    common_arg.add_argument('--config_save', type=str, required=True, help='Where to save the config')
    common_arg.add_argument('--vocab_save', type=str, help='Where to save the vocab')
    common_arg.add_argument('--vocab_load', type=str, help='Where to load the vocab; otherwise it will be evaluated')
    common_arg.add_argument('--model_load', type=str, help='Where to load the model')
    return parser
    
    
def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--num_layers", type=int, default=3,
                           help="Number of LSTM layers")
    model_arg.add_argument("--hidden", type=int, default=768,
                           help="Hidden size")
    model_arg.add_argument("--dropout", type=float, default=0.2,
                           help="dropout between LSTM layers except for last")

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=12,             # default 80
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,     # default 1
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,   # defualt 1
                           help='Number of workers for DataLoaders')
    parser = add_train_args(parser)
    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
