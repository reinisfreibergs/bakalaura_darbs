import copy
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
# import pre_processor_v2
import csv_result_parser as result_parser
import time
from datetime import datetime
from modules.csv_utils_2 import CsvUtils2
from modules.file_utils import FileUtils
from modules.args_utils import ArgsUtils


def param_count(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
    return total_param_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='model_1_LSTM', type=str)
    # parser.add_argument('-model', default='model_2_phased_lstm', type=str)
    # parser.add_argument('-model', default='model_3_PLSTM', type=str)
    parser.add_argument('-datasource', default='datasource_2', type=str)
    parser.add_argument('-sequence_name', default='test_run', type=str)
    parser.add_argument('-run_name', default='run', type=str)
    parser.add_argument('-id', default=0, type=int)
    parser.add_argument('-is_single_cuda_device', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('-learning_rate', default=1e-6, type=float)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-epochs', default=5, type=int)
    parser.add_argument('-hidden_size', default=16, type=int)
    parser.add_argument('-sequence_len', default=100, type=int)
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-train_size', default=0.8, type=int)
    parser.add_argument('-lstm_layers', default=2, type=int)
    parser.add_argument('-num_workers', default=0, type=int)

    parser.add_argument('-csv_directory', default='../data/original/dpc_dataset_csv', type=str) #'../data/original/dpc_dataset_csv'
    # parser.add_argument('-output_directory', default='./datasource', type=str)
    # parser.add_argument('-csv_directory', default='./dummy_csv', type=str) #'../data/original/dpc_dataset_csv'
    parser.add_argument('-output_directory', default='./datasource_v2', type=str)
    # parser.add_argument('-output_directory', default='D:/bakalaura_darbs', type=str)


    parser.add_argument('-early_stopping_patience', default=10, type=int)
    parser.add_argument('-early_stopping_param', default='test_loss', type=str)
    parser.add_argument('-early_stopping_delta_percent', default=0.1, type=float)

    args, args_other = parser.parse_known_args()
    args = ArgsUtils.add_other_args(args, args_other)
    path_sequence = f'./results/{args.sequence_name}'
    args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
    path_run = f'./results/{args.sequence_name}/{args.run_name}'
    path_artificats = f'./artifacts/{args.sequence_name}/{args.run_name}'
    FileUtils.createDir(path_run)
    FileUtils.createDir(path_artificats)
    FileUtils.writeJSON(f'{path_run}/{args.id}_args_{args.sequence_name}.json', vars(args))
    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, args.run_name)


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

    is_data_parallel = False
    if not args.is_single_cuda_device:
        if args.device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, dim=0)
            is_data_parallel = True

    get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']),
                               'get_data_loaders')
    dataset_train, dataset_test, max_value, min_value = get_data_loaders(args)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = args.learning_rate
    )


    metrics = {}
    best_metrics = {}
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
        # metrics_csv = []
        # metrics_csv.append(epoch)
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

                model_module = model
                if is_data_parallel:
                    model_module = model.module

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    best_metrics[f'{stage}_loss'] = value
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 10)}')
            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        plt.clf()
        plts = []
        c = 0
        # for key, value in metrics.items():
        #     metrics_csv.append(value[-1])

        min_train_loss = min(metrics['train_loss'])
        min_test_loss = min(metrics['test_loss'])
        best_metrics['min_train_loss'] = min_train_loss
        best_metrics['min_test_loss'] = min_test_loss
            # plts += plt.plot(value, f'C{c}', label=key)
            # ax = plt.twinx()
            # c += 1

        # plt.legend(plts, [it.get_label() for it in plts])
        # plt.draw()
        # plt.pause(0.1)

        if best_test_loss > loss.item():
            best_test_loss = loss.item()
            torch.save(model_module.cpu().state_dict(), f'{path_run}/model_test_{args.sequence_len}_test_loss.pt')
            model = model.to(args.device)

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

        CsvUtils2.add_hparams(
            path_sequence=path_sequence,
            run_name=args.run_name,
            args_dict=args.__dict__,
            metrics_dict=best_metrics,
            global_step=epoch
        )
