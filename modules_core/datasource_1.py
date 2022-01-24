import torch
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset_time_series(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.args = args

        self.data = np.memmap(
            filename=f'{args.output_directory}/{dataset_name}.mmap',
            dtype='float16',
            mode='r+',
            shape= memmap_shape
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_init = self.data[idx]
        x_init = torch.FloatTensor(x_init)

        y_init = torch.roll(input=x_init,shifts=-1,dims=0)
        x_1 = x_init[:SEQUENCE_LENGTH]
        y_1 = x_init[SEQUENCE_LENGTH+1:2*SEQUENCE_LENGTH+1]
        z_1 = x_init[2*SEQUENCE_LENGTH+2:3*SEQUENCE_LENGTH+2]
        k_1 = x_init[3*SEQUENCE_LENGTH+3:4*SEQUENCE_LENGTH+3]

        x_2 = y_init[:SEQUENCE_LENGTH]
        y_2 = y_init[SEQUENCE_LENGTH+1:2*SEQUENCE_LENGTH+1]
        z_2 = x_init[2*SEQUENCE_LENGTH+2:3*SEQUENCE_LENGTH+2]
        k_2 = x_init[3*SEQUENCE_LENGTH+3:4*SEQUENCE_LENGTH+3]

        x = torch.stack([x_1,y_1,z_1,k_1], dim=1) # [sin(theta1), cos(theta1), sin(theta2), cos(theta2)]
        y = torch.stack([x_2,y_2,z_2,k_2], dim=1)

        return x, y


def get_data_loaders(args):
    init_dataset = Dataset_time_series()
    subsetA, subsetB = train_test_split(init_dataset, train_size=args.train_size, shuffle=False )

    dataset_train = torch.utils.data.DataLoader(
        dataset = subsetA,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory = True,
        drop_last = True,
        num_workers = args.num_workers
    )

    dataset_test = torch.utils.data.DataLoader(
        dataset = subsetB,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory = True,
        drop_last = True,
        num_workers = args.num_workers
    )

    return dataset_train, dataset_test
