import copy
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
# import data_pre_processing
import csv_result_parser as result_parser
import time

def param_count(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
    return total_param_size

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-hidden_size', default=64, type=int)
parser.add_argument('-sequence_len', default=100, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-train_size', default=0.8, type=int)
parser.add_argument('-lstm_layers', default=2, type=int)

parser.add_argument('-csv_directory', default='../data/original/dpc_dataset_csv', type=str) #'../data/original/dpc_dataset_csv'
parser.add_argument('-output_directory', default='./datasource', type=str)

parser.add_argument('-early_stopping_patience', default=10, type=int)
parser.add_argument('-early_stopping_param', default='test_loss', type=str)
parser.add_argument('-early_stopping_delta_percent', default=1e-3, type=float)

args = parser.parse_args()


dataset_name = f'{args.sequence_len}_dataset'
if not os.path.exists(f'./datasource/{dataset_name}.json'):
    data_pre_processing.create_memmap_dataset(args)

with open(f'./datasource/{dataset_name}.json', 'r') as fp:
    memmap_info = json.load(fp)
memmap_shape = tuple(memmap_info['shape'])

class Dataset_time_series(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data = np.memmap(
            filename=f'./datasource/{dataset_name}.mmap',
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

SEQUENCE_LENGTH = args.sequence_len
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
SEQUENCE_LEN = args.sequence_len
DEVICE = args.device
EPOCHS = args.epochs
HIDDEN_SIZE = args.hidden_size

init_dataset = Dataset_time_series()
subsetA, subsetB = train_test_split(init_dataset, train_size=args.train_size, shuffle=False )

dataset_train = torch.utils.data.DataLoader(
    dataset = subsetA,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory = True
)

dataset_test = torch.utils.data.DataLoader(
    dataset = subsetB,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory = True
)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=HIDDEN_SIZE),
            torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        )
        self.lstm_layer = torch.nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True, num_layers=args.lstm_layers)
        self.linear_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE),
            torch.nn.Mish(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=4)
        )
    def forward(self, x):
        y_1 = self.linear_1.forward(x)
        lstm_out, _ = self.lstm_layer.forward(y_1)
        y_prim = self.linear_2.forward(lstm_out)

        return y_prim

model = Model().to(DEVICE)
parameter_count = param_count(model)
print(parameter_count)

optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr = LEARNING_RATE
)


metrics = {}
best_test_loss = float(9999)
for stage in ['train', 'test']:
    for metric in [
        'loss'
    ]:
        metrics[f'{stage}_{metric}'] = []


max_value = init_dataset.data.max()
min_value = init_dataset.data.min()

metric_before = {}
early_stopping_patience = 0

filename = result_parser.run_file_name()
for epoch in range(1, EPOCHS+1):
    start = time.time()
    metrics_csv = []
    metrics_csv.append(epoch)
    metric_mean = {}
    for data_loader in [dataset_train, dataset_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataset_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)

            loss = ( torch.sqrt( torch.mean((y - y_prim)**2) ) ) / ( max_value - min_value )

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
                metrics_strs.append(f'{key}: {round(value, 10)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        metrics_csv.append(value[-1])
        # plts += plt.plot(value, f'C{c}', label=key)
        # ax = plt.twinx()
        # c += 1

    # plt.legend(plts, [it.get_label() for it in plts])
    # plt.draw()
    # plt.pause(0.1)

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/model_test_{args.sequence_len}_single_epoch.pt')
        model = model.to(DEVICE)

    result_parser.run_csv(file_name=f'./results/{filename}_{str(args.learning_rate)}.csv',
                        metrics=metrics_csv)
    epoch_time = time.time() - start
    print(epoch_time)

        # early stopping
    percent_improvement = 0
    if epoch > 1:
        if metric_before[args.early_stopping_param] != 0:
            if np.isnan(metrics[args.early_stopping_param][-1]) or np.isinf(metrics[args.early_stopping_param][-1]):
                print('loss isnan break')
                break

            percent_improvement = -(metrics[args.early_stopping_param][-1] - metric_before[args.early_stopping_param][-1]) / \
                                  metric_before[args.early_stopping_param][-1]

            print(f' percent_improvement {percent_improvement}')
            if np.isnan(percent_improvement):
                percent_improvement = 0

            if metrics[args.early_stopping_param][-1] >= 0:
                if args.early_stopping_delta_percent > percent_improvement:
                    early_stopping_patience += 1
                else:
                    early_stopping_patience = 0
        if early_stopping_patience > args.early_stopping_patience:
            print('early_stopping_patience break')
            break
    metric_before = copy.deepcopy(metrics)

result_parser.best_result_csv(result_file='comparison_results.csv',
                            run_file=f'results/{filename}_{str(args.learning_rate)}.csv',
                            run_name=filename,
                            batch_size= args.batch_size,
                            learning_rate= args.learning_rate,
                            param_count=parameter_count)
