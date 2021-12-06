import matplotlib.pyplot as plt
import torch
import numpy as np
from csv import reader
import math
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import json
import os
import data_pre_processing



class Dataset_time_series(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.SEQUENCE_LENGTH = args.sequence_len
        self.args = args

        dataset_name = f'{args.sequence_len}_dataset'
        if not os.path.exists(f'./datasource/{dataset_name}.json'):
            data_pre_processing.create_memmap_dataset(args)

        with open(f'./datasource/{dataset_name}.json', 'r') as fp:
            memmap_info = json.load(fp)
        memmap_shape = tuple(memmap_info['shape'])

        self.data = np.memmap(
            filename=f'./datasource/{dataset_name}.mmap',
            dtype='float16',
            mode='r+',
            shape= memmap_shape #(9,8) - dummy, (375978, 202) - 100 seq length
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_init = self.data[idx]
        x_init = torch.FloatTensor(x_init)
        y_init = torch.roll(input=x_init,shifts=-1,dims=0)

        x_1 = x_init[:self.SEQUENCE_LENGTH]
        y_1 = x_init[self.SEQUENCE_LENGTH+1:2*self.SEQUENCE_LENGTH+1]
        z_1 = x_init[2*self.SEQUENCE_LENGTH+2:3*self.SEQUENCE_LENGTH+2]
        k_1 = x_init[3*self.SEQUENCE_LENGTH+3:4*self.SEQUENCE_LENGTH+3]

        x_2 = y_init[:self.SEQUENCE_LENGTH]
        y_2 = y_init[self.SEQUENCE_LENGTH+1:2*self.SEQUENCE_LENGTH+1]
        z_2 = x_init[2*self.SEQUENCE_LENGTH+2:3*self.SEQUENCE_LENGTH+2]
        k_2 = x_init[3*self.SEQUENCE_LENGTH+3:4*self.SEQUENCE_LENGTH+3]

        x = torch.stack([x_1,y_1,z_1,k_1], dim=1)
        y = torch.stack([x_2,y_2,z_2,k_2], dim=1)

        return x, y


def get_data_loaders(args):

    init_dataset = Dataset_time_series(args)
    if args.is_overfitting:
        subsetA = init_dataset
        subsetB = init_dataset
    else:
        subsetA, subsetB = train_test_split(init_dataset, test_size=0.2, shuffle=False )

    dataset_train = torch.utils.data.DataLoader(
        dataset = subsetA,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory = True
    )

    dataset_test = torch.utils.data.DataLoader(
        dataset = subsetB,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory = True
    )

    return dataset_train, dataset_test
