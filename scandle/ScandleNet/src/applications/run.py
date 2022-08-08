'''
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
'''

import sys, os
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict

from ScandleNet.src.core.data import get_data
from ScandleNet.src.applications.scandlenet import run_net

import tensorflow as tf

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='0')
parser.add_argument('--dset', type=str, help='gpu number to use', default='fashion_mnist')
args = parser.parse_args()

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
if args.dset == 'fashion_mnist':
    mnist_params = {
        'n_clusters': 10,
        'use_code_space': False,
        'affinity': 'ana',
        'n_nbrs': 3,
        'scale_nbr': 2,
        'siam_k': 2,
        'siam_ne': 1,
        'spec_ne': 100,
        'siam_lr': 1e-3,
        'spec_lr': 1e-3,
        'siam_patience': 10,
        'spec_patience': 20,
        'siam_drop': 0.1,
        'spec_drop': 0.1,
        'batch_size': 1024,
        'siam_reg': None,
        'spec_reg': None,
        'siam_n': None,
        'siamese_tot_pairs': 600000,
        'arch': [

            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 512},
            {'type': 'relu', 'size': 10},
            ],
        'use_approx': False,
        }
    params.update(mnist_params)


# LOAD DATA
data = get_data(params)

# RUN EXPERIMENT

x_scandlenet, y_scandlelnet = run_net(data, params)


