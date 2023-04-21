import glob
import os

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

import math
from pathlib import Path
import json
from collections import OrderedDict
from itertools import repeat
import pandas as pd

# Label dicts
pain_labels = {
    'BL1': 0,
    'PA1': 1,
    'PA2': 2,
    'PA3': 3,
    'PA4': 4
}

# Multiclass type
class_types_dict = {
    '0 vs 1 vs 2 vs 3 vs 4': 0,
    '0 vs 4': 1,
    '0 vs 1, 2, 3, 4': 2,
    '0, 1 vs 3, 4': 3,
    '0 vs 3, 4': 4,
    '0, 1 vs 4': 5,
    '0, 1 vs 3': 6,
    '0 vs 1, 2 vs 3, 4': 7
}


def generate_kfolds_index(npz_dir, k_folds) -> dict[int: list[str]]:
    """
    Generate k-folds dataset index and store into a dictionary. The length of dictionary is equal to the number of
    folds. Each element contains training set and testing set.
    :param npz_dir: npz files directory
    :param k_folds: the number of folds
    :return: a dict contains k-folds dataset paths, e.g. dict{0: [list[str(train_dir)], list[str(test_dir)]]..., k:[...]}
    """

    if os.path.exists(npz_dir):
        print('================= Creating KFolds Index =================')
    else:
        print('================= Data directory does not exist =================')

    npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
    npz_files = np.asarray(npz_files)
    kfolds_names = np.array_split(npz_files, k_folds)
    # print(kfolds_names)
    kfolds_index = {}
    for fold_index in range(0, k_folds):
        test_data = kfolds_names[fold_index].tolist()
        train_data = [files for i, files in enumerate(kfolds_names) if i != fold_index]
        train_data = [files for subfiles in train_data for files in subfiles]
        kfolds_index[fold_index] = [train_data, test_data]
    print('================= {} folds dataset created ================='.format(k_folds))
    return kfolds_index


class BioVidLoader(Dataset):
    """
    Input: a list of npz files' directories from k-folds index
    Output: a tensor of values and labels
    """
    def __init__(self, npz_files, label_converter):
        super(BioVidLoader, self).__init__()

        # Load first npz file which is easy to handle for the rest
        x_values = np.load(npz_files[0])['x']
        y_labels = np.load(npz_files[0])['y']

        # Load npz files starting from position 1
        for file in npz_files[1:]:
            x_values = np.vstack((x_values, np.load(file)['x']))
            y_labels = np.append(y_labels, np.load(file)['y'])

        # Convert the original labels to the task labels, e.g. 4 --> 1
        y_labels = np.array([label_converter[str(label)] for label in y_labels])
        # Create masking indices, -1 as False
        mask = (y_labels != -1)

        # Remove all the values with False
        x_values = x_values[mask]
        y_labels = y_labels[mask]

        self.val = torch.from_numpy(x_values).float()
        self.lbl = torch.from_numpy(y_labels).long()

        # Change shape to (Batch size, Channel size, Length)
        self.val = self.val.unsqueeze(1)

    def __len__(self):
        return self.val.shape[0]

    def __getitem__(self, idx):
        return self.val[idx], self.lbl[idx]

    def __repr__(self):
        return '{}'.format(repr(self.val))

    def __str__(self):
        # BioVid: torch.Size([3440, 1, 2816]), torch.Size([3440])
        return 'The shape of values and labels: {}, {}'.format(self.val.shape, self.lbl.shape)


def load_data(train_set, valid_set, label_converter, batch_size, num_workers = 0) -> tuple[DataLoader, DataLoader, list[int]]:
    """
    generate dataloader for both training dataset and validation dataset from one of the k-folds.
    :param train_set: training dataset
    :param valid_set: validation dataset
    :param label_converter: convert the original labels to the desired labels
    :param batch_size: batch size
    :param num_workers: 4*GPU
    :return: dataloader for training dataset, validation dataset, the number of samples for each class,
        e.g. two classes -> list[int,int]
    """
    train_dataset = BioVidLoader(train_set, label_converter)
    valid_dataset = BioVidLoader(valid_set, label_converter)

    cat_y = torch.cat((train_dataset.lbl, valid_dataset.lbl))

    # dist = [cat_y.count(i) for i in range(n_classes)]

    # e.g. two classes (tensor(list[lbl, lbl]), tensor(list[int, int]))
    unique_counts = cat_y.unique(return_counts = True)
    # number of samples for each class -> list[int, int]
    dist = unique_counts[1].tolist()

    # n_classes = len(unique_counts[0])

    # percent = [pop / sum(dist) for pop in dist]

    train_loader = DataLoader(train_dataset,
                              num_workers = num_workers,
                              batch_size = batch_size,
                              shuffle = True,
                              drop_last = False,
                              pin_memory = True)

    valid_loader = DataLoader(valid_dataset,
                              num_workers = num_workers,
                              batch_size = batch_size,
                              shuffle = False,
                              drop_last = False,
                              pin_memory = True)
    return train_loader, valid_loader, dist




