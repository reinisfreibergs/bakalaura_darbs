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
import csv_result_parser as result_parser

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-hidden_size', default=16, type=int)
parser.add_argument('-sequence_len', default=5, type=int)
parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-csv_directory', default='./dummy_csv', type=str) #'../data/original/dpc_dataset_csv'
parser.add_argument('-output_directory', default='./datasource', type=str)
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
            shape= memmap_shape #(9,8) - dummy, (375978, 202) - 100 seq length
        )

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

SEQUENCE_LENGTH = args.sequence_len
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
best_test_loss = float(9999)
for stage in ['train', 'test']:
    for metric in [
        'loss'
    ]:
        metrics[f'{stage}_{metric}'] = []

filename = result_parser.run_file_name()
for epoch in range(1, EPOCHS+1):
    metrics_csv = []
    metrics_csv.append(epoch)
    for data_loader in [dataset_train, dataset_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataset_test:
            stage = 'test'

        for x, y in tqdm(data_loader):

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

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        metrics_csv.append(value[-1])
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.draw()
    plt.pause(0.1)

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/model_test.pt')
        model = model.to(DEVICE)

    # torch.save(model.state_dict(), model_path)
    # save chekpoint
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss
    # }, model_path)

    result_parser.run_csv(file_name=f'results/{filename}',
                        metrics=metrics_csv)

result_parser.best_result_csv(result_file='11.1_comparison_results.csv',
                            run_file=f'results/{filename}',
                            run_name=filename,
                            batch_size= args.batch_size,
                            learning_rate= args.learning_rate)
