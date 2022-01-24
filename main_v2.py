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
# import pre_processor_v2
import csv_result_parser as result_parser
import time
from datetime import datetime

def param_count(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
    return total_param_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-model', default='model_1_LSTM', type=str)
    # parser.add_argument('-model', default='model_2_phased_lstm', type=str)
    parser.add_argument('-model', default='model_3_PLSTM', type=str)
    parser.add_argument('-datasource', default='datasource_2', type=str)

    parser.add_argument('-learning_rate', default=1e-6, type=float)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-epochs', default=100, type=int)
    parser.add_argument('-hidden_size', default=16, type=int)
    parser.add_argument('-sequence_len', default=100, type=int)
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-train_size', default=0.8, type=int)
    parser.add_argument('-lstm_layers', default=2, type=int)
    parser.add_argument('-num_workers', default=0, type=int)

    parser.add_argument('-csv_directory', default='../data/original/dpc_dataset_csv', type=str) #'../data/original/dpc_dataset_csv'
    # parser.add_argument('-output_directory', default='./datasource', type=str)
    # parser.add_argument('-csv_directory', default='./dummy_csv', type=str) #'../data/original/dpc_dataset_csv'
    # parser.add_argument('-output_directory', default='./datasource_v2', type=str)
    parser.add_argument('-output_directory', default='D:/bakalaura_darbs', type=str)


    parser.add_argument('-early_stopping_patience', default=10, type=int)
    parser.add_argument('-early_stopping_param', default='test_loss', type=str)
    parser.add_argument('-early_stopping_delta_percent', default=0.1, type=float)

    args = parser.parse_args()


    dataset_name = f'{args.sequence_len}_dataset'
    if not os.path.exists(f'{args.output_directory}/{dataset_name}.json'):
        pre_processor_v2.create_memmap_dataset(args)

    with open(f'{args.output_directory}/{dataset_name}.json', 'r') as fp:
        memmap_info = json.load(fp)
    memmap_shape = tuple(memmap_info['shape'])

    Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
    model = Model(args).to(args.device)
    parameter_count = param_count(model)
    print(parameter_count)

    get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']),
                               'get_data_loaders')
    dataset_train, dataset_test, max_value, min_value = get_data_loaders(args)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = args.learning_rate
    )


    metrics = {}
    best_test_loss = float(9999)
    for stage in ['train', 'test']:
        for metric in [
            'loss'
        ]:
            metrics[f'{stage}_{metric}'] = []

    metric_before = {}
    early_stopping_patience = 0

    filename = result_parser.run_file_name()
    for epoch in range(1, args.epochs+1):
        start = time.time()
        metrics_csv = []
        metrics_csv.append(epoch)
        metric_mean = {}
        for data_loader in [dataset_train, dataset_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            if data_loader == dataset_test:
                stage = 'test'

            for x, y in tqdm(data_loader):

                x = x.to(args.device)
                y = y.to(args.device)

                y_prim = model.forward(x)

                # loss = ( torch.sqrt( torch.mean((y - y_prim)**2) ) ) / ( max_value - min_value )
                # loss = torch.mean( torch.sqrt(torch.abs(y - y_prim) + 1e-8 ) )
                # loss = torch.mean(torch.abs(y-y_prim))
                loss = torch.mean((y-y_prim)**2)

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
            torch.save(model.cpu().state_dict(), f'./results/model_test_{args.sequence_len}_{datetime.utcnow().strftime(f"%y-%m-%d--%H-%M-%S")}.pt')
            model = model.to(args.device)

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

                percent_improvement = -((metrics[args.early_stopping_param][-1] - metric_before[args.early_stopping_param][-1]) / \
                                      metric_before[args.early_stopping_param][-1])*100

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
