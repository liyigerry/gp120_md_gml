"""
PyTorch compatible dataloader
"""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
import pickle

# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    des, labels = map(list, zip(*samples))
    batched_des = dgl.batch(des)
    labels = torch.tensor(labels)
    return batched_des,labels

class GruDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=collate,
                 seed=0,
                 shuffle=True,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}

        labels =[l for des,l in dataset]

        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle)
        elif split_name == 'fold5':
            train_idx, valid_idx = self._split_fold5(
                labels, fold_idx, seed, shuffle)
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size, **self.kwargs)
        self.valid_loader = DataLoader(
            dataset, sampler=valid_sampler,
            batch_size=batch_size, **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        ''' 10 flod '''
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 10.")

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print("train_set : test_set = {} : {}".format(
            len(train_idx), len(valid_idx)))

        return train_idx, valid_idx
    def _split_fold5(self, labels, fold_idx=0, seed=0, shuffle=True):
        ''' 5 flod '''
        assert 0 <= fold_idx and fold_idx < 5, print(
            "fold_idx must be from 0 to 5.")

        skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print("train_set : test_set = {} : {}".format(
            len(train_idx), len(valid_idx)))

        return train_idx, valid_idx
