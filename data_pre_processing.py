import numpy as np
from csv import reader
import math
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-sequence_len', default=4, type=int)
args = parser.parse_args()

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

seq_len = args.sequence_len
directory = '../data/original/dpc_dataset_csv'

# df = pd.read_csv('lengths.csv', header=None)
# sum_csv_lengths = sum(df.values[0])
# number_of_csv_files = len(df.values[0])
#
# data_shape = (sum_csv_lengths - number_of_csv_files * seq_len, 2*seq_len)
#
# dataset = np.memmap(
#     filename='dataset.mmap',
#     dtype='float16',
#     mode = 'r+',
#     shape=(amount_of_samples, 2*seq_len)
# )

x,y = prepare_training_data(file='0_test_10.csv', sequence_len=seq_len)
x_np = np.array(x)
amount_of_samples = x_np.shape[0]

test_mmap = np.memmap(filename='test.mmap',dtype='float16', mode='w+', shape=(amount_of_samples,2*seq_len))
for i in range(amount_of_samples):
    for n in range(seq_len):
        test_mmap[i,n] = x_np[i,n,0]
        test_mmap[i, (n+seq_len)] = x_np[i,n,1]
test_mmap.flush()

print(test_mmap)


