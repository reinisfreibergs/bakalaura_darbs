import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json

class Dataset_time_series(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dataset_name = f'{args.sequence_len}_dataset'

        with open(f'{args.output_directory}/{dataset_name}.json', 'r') as fp:
            memmap_info = json.load(fp)
            memmap_shape = tuple(memmap_info['shape'])


        self.data = np.memmap(
            filename=f'{self.args.output_directory}/{dataset_name}.mmap',
            dtype='float16',
            mode='r+',
            shape= memmap_shape
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_init = self.data[idx]
        x_init = torch.FloatTensor(x_init)

        x = x_init[:-1] #all but excluding last one
        y = x_init[1:] #all but the first one

        return x, y


def get_data_loaders(args):
    init_dataset = Dataset_time_series(args)
    subsetA, subsetB = train_test_split(init_dataset, train_size=args.train_size, shuffle=True, random_state=1)

    max_value = init_dataset.data.max()
    min_value = init_dataset.data.min()

    dataset_train = torch.utils.data.DataLoader(
        dataset = subsetA,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory = True,
        num_workers = args.num_workers
    )

    dataset_test = torch.utils.data.DataLoader(
        dataset = subsetB,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory = True,
        num_workers = args.num_workers
    )

    return dataset_train, dataset_test, max_value, min_value
