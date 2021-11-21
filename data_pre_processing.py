import numpy as np
from csv import reader
import math
import argparse
import os
import pandas as pd
import json


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
    window = []
    angles = coordinates_to_sin_cos(file)
    for row in angles:
        if len(window) < sequence_len:
            window.append(row)
        else:
            target_row = row
            x_data.append(window.copy())
            window.pop(0)
            window.append(target_row)

    return x_data

def create_memmap_dataset(args):
    FULL_SEQUENCE_LEN = args.sequence_len + 1

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # iterate trough all csv files and find lengths
    lengths = []
    for csv in os.listdir(args.csv_directory):
        csv_path = os.path.join(args.csv_directory, csv)
        csv_length = pd.read_csv(csv_path).shape[0]
        lengths.append(csv_length+1)

    #calculate mmap size
    sum_csv_lengths = sum(lengths)
    number_of_csv_files = len(lengths)
    data_shape = (sum_csv_lengths - number_of_csv_files * FULL_SEQUENCE_LEN, 2*FULL_SEQUENCE_LEN)

    data_mmap = np.memmap(
        filename=f'{args.output_directory}/{args.sequence_len}_dataset.mmap',
        dtype='float16',
        mode='w+',
        shape=data_shape
    )

    #make combinations of coordinates from every csv and write into memmap
    number = 0
    for csv_file in os.listdir(args.csv_directory):
        x = prepare_training_data(file= os.path.join(args.csv_directory,csv_file), sequence_len=FULL_SEQUENCE_LEN)
        x_np = np.array(x)
        amount_of_samples = x_np.shape[0]

        for i in range(amount_of_samples):
            for n in range(FULL_SEQUENCE_LEN):
                data_mmap[number+i, n] = x_np[i,n,0]
                data_mmap[number+i, n+FULL_SEQUENCE_LEN] = x_np[i,n,1]

        number += amount_of_samples

    data_mmap.flush()
    print(data_shape)

    data_description = {'shape': data_shape}
    with open(f'{args.output_directory}/{args.sequence_len}_dataset.json', "w+") as description_file:
        json.dump(data_description, description_file)
