import models
import torch.nn as nn
import torch
import numpy as np
import datetime
import socket
import json
import argparse
import data_preprocessing
import random
import os
import math
import copy
import utils
import pandas as pd

import training_rnn
import training_ae

dt_object = datetime.datetime.now()
parser = argparse.ArgumentParser()

#LSTM args
# parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
# parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
# parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
# parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
# parser.add_argument('--nb_epoch', help='training iterations', default=400, type=int)
# parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=2560, type=int)
# parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=2560,
#                     type=int)
# parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
# parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks',
#                     type=str)  # we would like to end up with some non-stochastic & at least pseudo likelihood metric
# parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
# parser.add_argument('--dropout_prob', help='dropout_prob', default=0.3, type=float)
# parser.add_argument('--training_learning_rate', help='GD learning rate', default=1e-4, type=float)
# parser.add_argument('--training_gaussian_process', help='GP', default=1e-5, type=float)
# parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
# parser.add_argument('--dataset', help='dataset', default='', type=str)
# parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
# parser.add_argument('--random', help='if random', default=True, type=bool)
# parser.add_argument('--gpu', help='gpu', default=0, type=int)
# parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!',
#                     default='[0,1,4,10,15]', type=str)
# parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
# parser.add_argument('--architecture', help='BERT or GPT', default='BERT', type=str)
# parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
# parser.add_argument('--device', help='GPU or CPU', default='GPU', type=str)
# parser.add_argument('--lagrange_a', help='Langrange multiplier', default=1.0, type=float)
# parser.add_argument('--save_criterion_threshold', help='save_criterion_threshold', default=4.0, type=float)
# parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
# parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
# parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=1.0, type=float)
# parser.add_argument('--early_stopping', help='early_stopping', default=True, type=bool)
# parser.add_argument('--single_position_target', help='single_position_target', default=True, type=bool)




#AE args
parser.add_argument('--datetime', help='datetime', default=dt_object.strftime("%Y%m%d%H%M"), type=str)
parser.add_argument('--hidden_dim', help='hidden state dimensions', default=128, type=int)
parser.add_argument('--n_layers', help='number of layers', default=4, type=int)
parser.add_argument('--n_heads', help='number of heads', default=4, type=int)
parser.add_argument('--nb_epoch', help='training iterations', default=400, type=int)
parser.add_argument('--training_batch_size', help='number of training samples in mini-batch', default=1536, type=int)
parser.add_argument('--validation_batch_size', help='number of validation samples in mini-batch', default=1536,
                    type=int)
parser.add_argument('--training_mlm_method', help='training MLM method', default='BERT', type=str)
parser.add_argument('--validation_mlm_method', help='validation MLM method', default='fix_masks',
                    type=str)  # we would like to end up with some non-stochastic & at least pseudo likelihood metric
parser.add_argument('--mlm_masking_prob', help='mlm_masking_prob', default=0.15, type=float)
parser.add_argument('--dropout_prob', help='dropout_prob', default=0.3, type=float)
parser.add_argument('--training_learning_rate', help='GD learning rate', default=1e-4, type=float)
parser.add_argument('--training_gaussian_process', help='GP', default=1e-5, type=float)
parser.add_argument('--validation_split', help='validation_split', default=0.2, type=float)
parser.add_argument('--dataset', help='dataset', default='', type=str)
parser.add_argument('--random_seed', help='random_seed', default=1982, type=int)
parser.add_argument('--random', help='if random', default=True, type=bool)
parser.add_argument('--gpu', help='gpu', default=0, type=int)
parser.add_argument('--validation_indexes', help='list of validation_indexes NO SPACES BETWEEN ITEMS!',
                    default='[0,1,4,10,15]', type=str)
parser.add_argument('--ground_truth_p', help='ground_truth_p', default=0.0, type=float)
parser.add_argument('--architecture', help='BERT or GPT', default='BERT', type=str)
parser.add_argument('--time_attribute_concatenated', help='time_attribute_concatenated', default=False, type=bool)
parser.add_argument('--device', help='GPU or CPU', default='GPU', type=str)
parser.add_argument('--lagrange_a', help='Langrange multiplier', default=1.0, type=float)
parser.add_argument('--save_criterion_threshold', help='save_criterion_threshold', default=4.0, type=float)
parser.add_argument('--pad_token', help='pad_token', default=0, type=int)
parser.add_argument('--to_wrap_into_torch_dataset', help='to_wrap_into_torch_dataset', default=True, type=bool)
parser.add_argument('--seq_ae_teacher_forcing_ratio', help='seq_ae_teacher_forcing_ratio', default=0.9, type=float)
parser.add_argument('--early_stopping', help='early_stopping', default=True, type=bool)
parser.add_argument('--single_position_target', help='single_position_target', default=False, type=bool)


#experiment configs
epochs=250
no_configs=1000

#Parser configs
dropout_prob_range=[0, 1]
batch_size_range = [1, 5120]
learning_rate_range = [0, 1]
gaussian_process_range = [0, 1]

layer_no_range = [2,6]
#layer_size_range = [8,256]
#heads_no_range = [1,5]

lagrange_range_low = [0, 1]
lagrange_range_high = [1, 10]

# Max Trainable PARAMETERS
max_size=1000000
parser.add_argument('--max_parameters', help='maximum number of trainable parameters', default=max_size, type=float)

for i in range(0, no_configs):

    #LAYERS
    #layer_size = random.uniform(layer_size_range[0], layer_size_range[1])
    nlayers = random.uniform(layer_no_range[0], layer_no_range[1])
    #nheads = random.uniform(heads_no_range[0], heads_no_range[1])
    layer_size = 1
    nlayers = int(np.round(nlayers))
    #nheads = int(np.round(nheads))

    parser.set_defaults(hidden_dim=layer_size)
    parser.set_defaults(n_layers=nlayers)
    #parser.set_defaults(n_heads=nheads)

    #LAGRANGE
    if random.choice([0, 1]) == 0:
        parser.set_defaults(lagrange_a=random.uniform(lagrange_range_low[0], lagrange_range_low[1]))
    else:
        parser.set_defaults(lagrange_a=random.uniform(lagrange_range_high[0], lagrange_range_high[1]))

    #DROPOUT
    dropout_prob = random.uniform(dropout_prob_range[0], dropout_prob_range[1])
    parser.set_defaults(dropout_prob=dropout_prob)
    dropout_prob = random.uniform(dropout_prob_range[0], dropout_prob_range[1])

    #EPOCHS
    parser.set_defaults(nb_epoch=epochs)

    #LR AND GAUSSIAN
    learning_rate = random.uniform(learning_rate_range[0], learning_rate_range[1])
    gaussian = random.uniform(gaussian_process_range[0], gaussian_process_range[1])
    parser.set_defaults(training_learning_rate=learning_rate)
    parser.set_defaults(training_gaussian_process=gaussian)

    #BATCHSIZE
    batch_size = random.uniform(batch_size_range[0], batch_size_range[1])
    batch_size = int(np.round(batch_size))
    parser.set_defaults(training_batch_size=batch_size)
    parser.set_defaults(validation_batch_size=batch_size)

    dt_object = datetime.datetime.now()
    parser.set_defaults(datetime=dt_object.strftime("%Y%m%d%H%M"))
    args = parser.parse_args()
    training_ae.main(args, dt_object)
