import matplotlib.pyplot as plt
import torch
import numpy as np
from csv import reader
import math
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-hidden_size', default=16, type=int)
parser.add_argument('-sequence_len', default=100, type=int)
parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-mmap_filename', default='dataset.mmap', type=str)
args = parser.parse_args()
SEQUENCE_LENGTH = args.sequence_len
def raw_cartesian_to_polar_angles(l):
    '''Convert the cartesian coordinates to polar coordinates.'''
    x_red, y_red, x_green, y_green, x_blue, y_blue = [int(x) for x in l]

    angle_green_red = math.atan2((y_green-y_red),(x_green-x_red))
    angle_blue_green = math.atan2((y_blue-y_green),(x_blue-x_green))

    return [np.sin(angle_green_red), np.cos(angle_blue_green)]

def coordinates_to_sin_cos(file):
    angles = []
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        coordinates = list(csv_reader)

    for row in coordinates:
        sin_cos = raw_cartesian_to_polar_angles(row)
        angles.append([sin_cos[0], sin_cos[1]])

    return angles


def prepare_training_data(file, sequence_len):
    x_data = []
    y_data = []
    window = []
    angles = coordinates_to_sin_cos(file)
    for row in angles:
        if len(window) < sequence_len:
            window.append(row)
        else:
            target_row = row

            x_data.append(window.copy())
            y_data.append(target_row.copy())

            window.pop(0)
            window.append(target_row)

    return x_data, y_data

class Dataset_time_series(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data = np.memmap(filename='./datasource/dataset.mmap', dtype='float16', mode='r+', shape= (375978, 202)) #(9,8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_init = self.data[idx]
        x_init = torch.FloatTensor(x_init)
        y_init = torch.roll(input=x_init,shifts=-1,dims=0)

        x_1 = x_init[:SEQUENCE_LENGTH]
        y_1 = x_init[SEQUENCE_LENGTH:-2]

        x_2 = y_init[:SEQUENCE_LENGTH]
        y_2 = y_init[SEQUENCE_LENGTH:-2]

        x = torch.stack([x_1,y_1], dim=1)
        y = torch.stack([x_2,y_2], dim=1)

        return x, y

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
SEQUENCE_LEN = args.sequence_len
DEVICE = args.device
EPOCHS = args.epochs
HIDDEN_SIZE = args.hidden_size


init_dataset = Dataset_time_series()
subsetA, subsetB = train_test_split(init_dataset, test_size=0.2, shuffle=False )

dataset_train = torch.utils.data.DataLoader(
    dataset = subsetA,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataset_test = torch.utils.data.DataLoader(
    dataset = subsetB,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=HIDDEN_SIZE),
            torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        )
        self.lstm_layer = torch.nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.linear_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE),
            torch.nn.Mish(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=2)
        )
    def forward(self, x):
        y_1 = self.linear_1.forward(x)
        lstm_out, _ = self.lstm_layer.forward(y_1)
        y_prim = self.linear_2.forward(lstm_out)

        return y_prim

model = Model().to(DEVICE)
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr = LEARNING_RATE
)


metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):
    # metrics_csv = []
    # metrics_csv.append(epoch)
    for data_loader in [dataset_train, dataset_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataset_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)

            loss = torch.mean((y - y_prim)**2)

            metrics_epoch[f'{stage}_loss'].append(loss.item())

            if data_loader == dataset_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 5)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')
