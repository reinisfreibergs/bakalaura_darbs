import matplotlib.pyplot as plt
import torch
import numpy as np
from csv import reader
import math
from sklearn.model_selection import train_test_split


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
    def __init__(self, X, Y):
        super().__init__()

        self.data = list(zip(X,Y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_idx, y_idx = self.data[idx]

        x_idx = torch.FloatTensor(x_idx)
        y_idx = torch.FloatTensor(y_idx)

        return x_idx, y_idx

BATCH_SIZE = 3
LEARNING_RATE = 1e-4
SEQUENCE_LEN = 4
DEVICE = 'cpu'
EPOCHS = 1000
HIDDEN_SIZE = 16

X,Y = prepare_training_data(file='0.csv', sequence_len=SEQUENCE_LEN)
init_dataset = Dataset_time_series(X,Y)
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
            print(f'x {x}')
            print(f'y {y}')
            print(f'y_prim {y_prim}')
            exit()
            loss = -torch.mean((y - y_prim)**2)

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
                metrics_strs.append(f'{key}: {round(value, 2)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')
