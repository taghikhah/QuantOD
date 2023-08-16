from __future__ import print_function

import argparse
from pathlib import Path

__all__ = ['get_args']


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  


def get_args():
    parser = argparse.ArgumentParser(description='Experiments with Normalizing Flows.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training Settings
    parser.add_argument('--epochs', '-e', type=int, default=25, 
                        help='Number of epochs to train.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=9e-5, 
                        help='The initial learning rate.')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, 
                        help='Batch size.')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-6, 
                        help='Weight decay (L2 penalty) value for optimizers.')
    parser.add_argument('--eps', '-ep', type=float, default=1e-4,
                        help='Epsilon value for Adam optimizer.')
    parser.add_argument('--betas', '-be', type=float, nargs='+', default=[0.8, 0.8],
                        help='Betas value for Adam optimizer.')
    parser.add_argument('--max-norm', '-mn', type=float, default=1.0,
                        help='Maximum norm of the gradients for clipping.')
    parser.add_argument('--q-nll', '-qn', type=float, default=0.05,
                        help='q-NLL Loss, where 0.0: min, 1.0: mean, and 0.05: 5-th quantile.')
    
    # Experiment Settings
    parser.add_argument('--seed', '-s', type=int, default=1688686000,
                        help='Random seed. Set -1 for random seed.')
    parser.add_argument('--gpus', '-g', type=int, default=1, 
                        help='Number of GPUs where 0 is using CPU.')
    parser.add_argument('--no-cuda', '-nc', action='store_true', default=False,
                        help='disables CUDA training and uses CPU instead')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Prints the training progress.')
    parser.add_argument('--save-path', '-sp', type=str, default=ROOT / 'experiments', 
                        help='Path to save the experiment.')
    parser.add_argument('--weight-path', '-wp', type=str, default=ROOT / 'weights',
                        help='Path to save the weights.')
    parser.add_argument('--save-freq', '-sf', type=int, default=5,
                        help='Frequency of saving the model.')
    parser.add_argument('--experiment-count', '-ec', type=int, default=3,
                        help='Number of test experiments to run.')
    parser.add_argument('--evalutaion-methods', '-em', type=str, nargs='+', default=['QuantOD'],
                        choices=['Mahalanobis', 'QuantOD', 'MSP', 'ODIN', 'Energy', 'GEM', 'OE'],
                        help='Choose the evaluation methods.')
    
    # Classifier Settings
    parser.add_argument('--classifier-model', '-cm', type=str, default='wideresnet40',
                        choices=['wideresnet40', 'mobilenetv2','resnet18', 'densenet121'],
                        help='Choose the classifier.')
    parser.add_argument('--classifier-weight', '-cw', type=str, default= '',
                        help='Absolute path to load the classifier weight.')
    parser.add_argument('--classifier-pretrained', '-cp', action='store_true', default=False,
                        help='Enables the use of pretrained classifier from Energy, VOS, and OE.')
    
    # Dataset Settings
    parser.add_argument('--data-path', '-dp', type=str, default=ROOT / 'data',
                        help='Path to download the choosen dataset.')
    parser.add_argument('--id-dataset', '-id', type=str, default='cifar10', 
                        choices=['cifar10','cifar100', 'cifar-5', 'cifar-10', 'cifar-25', 'cifar-50'],
                        help='Choose the ID dataset. Note that cifar-10 is a subset of cifar-100.')
    parser.add_argument('--ood-dataset', '-ood', type=str, default='', 
                        choices=['svhn', 'lsunr', 'lsunc', 'isun', 'textures', 'places365', 'all'],
                        help='Choose the OOD dataset (Test Only).')
    parser.add_argument('--shuffle-data', '-sd', action='store_true', default=False,
                        help='Shuffle the trainset of the dataset.')
    parser.add_argument('--num-workers', '-nw', type=int, default=1, 
                        help='Maximum number of workers for pre-fetching threads.')
    parser.add_argument('--validation-split', '-vs', type=float, default=0.2,
                        help='Validation split percentage.')
    parser.add_argument('--normalize-data', '-nd', action='store_true', default=False,
                        help='Normalize transformations for the dataset.')
    parser.add_argument('--image-size', '-is', type=int, default=32,
                        help='Image size of the dataset.')
    parser.add_argument('--resize-image', '-ri', action='store_true', default=False,
                        help='Resize the image to the image size of the dataset.')
    parser.add_argument('--no-prelims', '-np', action='store_true', default=False,
                        help='Disables the use of preliminary experiments.')
        
    # Flows Model Settings
    parser.add_argument('--flows-weight', '-fw', type=str, default='',
                        help='Absolute path to load the flow model weights.')
    parser.add_argument('--flows-model', '-fm', type=str, default='glow', 
                        choices=['realnvp', 'glow', 'nice', 'gin'],
                        help='The flow model (architecture) for coupling blocks.')
    parser.add_argument('--flows-clamp', '-fc', type=float, metavar='C', default=3.0, 
                        help='Clamping (amplification) value for coupling blocks.')
    parser.add_argument('--flows-steps', '-fs', type=int, metavar='S', default=8, 
                        help='Number of steps (coupling blocks) in the flow model.') 
    parser.add_argument('--flows-layers', '-fl', type=int, metavar='L', default=2,
                        help='Number of layers in coupling networks.')
    parser.add_argument('--flows-hidden', '-fh', type=int, metavar='H', default=512, 
                        help='Number of hidden units (neurons) for coupling networks.')
    parser.add_argument('--flows-dropout', '-fd', type=float, metavar='D', default=0.3, 
                        help='Dropout rate for coupling networks.')
    
    
    args = parser.parse_args()
    
    return args
