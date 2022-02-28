import copy
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
import pre_processor_v3
# import csv_result_parser as result_parser
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

L1, L2 = 0.091, 0.07
m1, m2 = 0.01, 0.01
g = 9.81

def deriv(theta1, z1, theta2, z2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    c, s = torch.cos(theta1-theta2), torch.sin(theta1-theta2)

    z1dot = (m2*g*torch.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*torch.sin(theta1)) / L1 / (m1 + m2*s**2)
    z2dot = ((m1+m2)*(L1*z1**2*s - g*torch.sin(theta2) + g*torch.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)

    return torch.stack((z1dot, z2dot), dim=2)

def calc_E(x):
    """Return the total energy of the system."""
    theta1, omega1, theta2, omega2 = x.T

    V = -(m1+m2)*L1*g*np.cos(theta1) - m2*L2*g*np.cos(theta2)
    T = 0.5*m1 * L1**2 * omega1**2 + 0.5*m2*(L1**2 * omega1**2 + L2**2 * omega2**2 +
            2*L1*L2*omega1*omega2*np.cos(theta1-theta2))

    return T + V

def ode_loss(y_prim, x):

    theta1, theta2 = y_prim[:,:,0], y_prim[:,:,1]

    omegas = torch.autograd.grad(outputs=y_prim, inputs=x, grad_outputs=torch.ones_like(y_prim), create_graph=True)[0]
    epsilons = torch.autograd.grad(outputs=omegas, inputs=x, grad_outputs=torch.ones_like(y_prim), create_graph=True)[0]

    epsilons_derived = deriv(theta1, omegas[:,:,0], theta2, omegas[:,:,1])
    return torch.mean((epsilons-epsilons_derived)**2)


def loss_rollout(x, hidden_vect):
    angles = torch.FloatTensor().to(args.device)
    start = x
    with torch.no_grad():
        for i in range(args.rollout_length):
            angles_current, hidden_vect = model.forward(start, hidden_vect)
            angles = torch.cat(tensors=(angles, angles_current[:,-1,:].unsqueeze(dim=0)), dim=1)
            start = angles_current[:,-1,:].unsqueeze(dim=1)

    loss = torch.mean(torch.abs(angles_current)) #difference between 0 across all batches
    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-model', default='model_1_LSTM', type=str)
    # parser.add_argument('-model', default='model_2_phased_lstm', type=str)
    # parser.add_argument('-model', default='model_3_PLSTM', type=str)
    # parser.add_argument('-model', default='model_4_snake_LSTM', type=str)
    parser.add_argument('-model', default='model_5_hidden', type=str)
    parser.add_argument('-datasource', default='datasource_2', type=str)
    parser.add_argument('-sequence_name', default='v3_first_run_ode', type=str)
    parser.add_argument('-run_name', default='run', type=str)
    parser.add_argument('-id', default=0, type=int)
    parser.add_argument('-is_single_cuda_device', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-epochs', default=100, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-sequence_len', default=400, type=int)
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-train_size', default=0.8, type=int)
    parser.add_argument('-lstm_layers', default=2, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-rollout_length', default=500, type=int)

    parser.add_argument('-activation', default='snake', type=str)
    parser.add_argument('-maxout_layers', default=2, type=int)

    parser.add_argument('-csv_directory', default='../data/original/dpc_dataset_csv', type=str)
    parser.add_argument('-output_directory', default='D:/bakalaura_darbs/v3', type=str)

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
        pre_processor_v3.create_memmap_dataset(args)

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

    for epoch in range(1, args.epochs+1):
        start = time.time()
        metric_mean = {}
        for data_loader in [dataset_train, dataset_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            model = model.train()
            torch.set_grad_enabled(True)
            if data_loader == dataset_test:
                stage = 'test'
                model = model.eval()

            for x, y in tqdm(data_loader):

                x = x.to(args.device)
                y = y.to(args.device)

                x.requires_grad = True
                with torch.backends.cudnn.flags(enabled=False):
                    y_prim, hidden = model.forward(x)

                loss_pred = torch.mean((y-y_prim)**2)
                loss_ode = ode_loss(y_prim, x)
                # loss_roll = loss_rollout(y_prim, hidden)
                loss = loss_pred + loss_ode

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


        min_train_loss = min(metrics['train_loss'])
        min_test_loss = min(metrics['test_loss'])
        best_metrics['min_train_loss'] = min_train_loss
        best_metrics['min_test_loss'] = min_test_loss


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
