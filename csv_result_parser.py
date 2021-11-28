import os.path
import sys
import os
from datetime import datetime
from csv import writer
import pandas as pd




def run_file_name():
    file_name =  str(os.path.basename(sys.argv[0]))  + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S')
    file_name_csv = str(file_name) + ".csv"
    return file_name_csv

def run_csv(file_name, metrics):
    file_exists = os.path.isfile(file_name)
    header = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']

    with open(file_name, 'a', newline='') as f_object:
        writer_object = writer(f_object)

        if not file_exists:
            writer_object.writerow(header)

        writer_object.writerow(metrics)

def best_result_csv(result_file, run_file,run_name, batch_size, learning_rate, transformer_heads, transformer_layers):

    read_file = pd.read_csv(run_file)
    max_epochs = read_file['epoch'].max()
    min_train_loss = read_file['train_loss'].min()
    max_train_acc = read_file['train_acc'].max()
    min_test_loss = read_file['test_loss'].min()
    max_test_acc = read_file['test_acc'].max()

    result_row = [run_name,
                  max_epochs,
                  batch_size,
                  learning_rate,
                  transformer_heads,
                  transformer_layers,
                  min_train_loss,
                  max_train_acc,
                  min_test_loss,
                  max_test_acc ]

    file_exists = os.path.isfile(result_file)
    headers = ['file','epochs','batch size', 'learning rate', 'transformer_heads', 'transformer_layers', 'min_train_loss', 'max_train_acc', 'min_test_loss', 'max_test_acc']

    with open(result_file, 'a', newline='') as result_object:
        writer_object = writer(result_object)
        if not file_exists:
            writer_object.writerow(headers)
        writer_object.writerow(result_row)


