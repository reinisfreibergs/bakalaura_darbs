import numpy as np
from csv import reader
import math
import argparse
import os
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

def raw_cartesian_to_polar_angles(l):
    '''Convert the cartesian coordinates to polar coordinates.'''
    x_red, y_red, x_green, y_green, x_blue, y_blue = [int(x) for x in l]

    angle_green_red = math.atan2((y_green-y_red),(x_green-x_red))
    angle_blue_green = math.atan2((y_blue-y_green),(x_blue-x_green))
    if angle_green_red < 0 :
        angle_green_red = 2*math.pi + angle_green_red # change range from [-pi, pi] to [0, 2pi]
    if angle_blue_green < 0:
        angle_blue_green = 2*math.pi + angle_blue_green

    return [angle_green_red, angle_blue_green]

def coordinates_to_sin_cos(file):
    angles = []
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        coordinates = list(csv_reader)

    for row in coordinates:
        # sin_cos = raw_cartesian_to_polar_angles(row)
        angles.append(raw_cartesian_to_polar_angles(row))
        # print(sin_cos[:5])
        # angles.append([sin_cos[0], sin_cos[1], sin_cos[2], sin_cos[3]])
        # print(angles[:5])
        # exit()

    return angles


def prepare_training_data(file, sequence_len):
    x_data = []
    window = []
    angles = np.array(coordinates_to_sin_cos(file)) #[0;2pi]
    angles_next = np.roll(angles, shift=1, axis=0)
    initial_diff = angles_next - angles

    initial_diff[initial_diff<-1] = angles_next[initial_diff<-1] - (angles[initial_diff<-1] - 2*math.pi)
    initial_diff[initial_diff>1] = (angles_next[initial_diff>1] - 2*math.pi) - angles[initial_diff>1]

    usable_diff = initial_diff[1:]
    usable_angles_next = angles_next[1:]

    angle_package = np.concatenate((usable_diff, usable_angles_next), axis=1)
    # angles_next = np.roll(angles, shift=1, axis=0)
    # diff = (angles_next - angles)[1:]
    for row in angle_package:
        if len(window) < sequence_len:
            window.append(row)
        else:
            target_row = row
            x_data.append(window.copy())
            window.pop(0)
            window.append(target_row)

    return x_data

def create_memmap_dataset(args):
    dataset_name = f'{args.sequence_len}_dataset'
    if not os.path.exists(f'{args.output_directory}/{dataset_name}.json'):
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
        # data_shape = (sum_csv_lengths - number_of_csv_files * FULL_SEQUENCE_LEN, 4*FULL_SEQUENCE_LEN)
        data_shape = (sum_csv_lengths - number_of_csv_files * FULL_SEQUENCE_LEN, FULL_SEQUENCE_LEN, 4)

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

            for i in tqdm(range(amount_of_samples)):
                for n in range(FULL_SEQUENCE_LEN):
                    data_mmap[number+i, n,:] = x_np[i,n,:]

            number += amount_of_samples

        data_mmap.flush()
        print(data_shape)

        data_description = {'shape': data_shape}
        with open(f'{args.output_directory}/{args.sequence_len}_dataset.json', "w+") as description_file:
            json.dump(data_description, description_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('-sequence_len', default=10, type=int)
    parser.add_argument('-csv_directory', default='../data/original/dpc_dataset_csv', type=str) #'../data/original/dpc_dataset_csv'
    # parser.add_argument('-output_directory', default='./datasource', type=str)
    # parser.add_argument('-csv_directory', default='./dummy_csv_2', type=str) #'../data/original/dpc_dataset_csv'
    parser.add_argument('-output_directory', default='./datasource_dummy', type=str)
    # parser.add_argument('-output_directory', default='D:/bakalaura_darbs/v3', type=str)


    args, args_other = parser.parse_known_args()

    create_memmap_dataset(args)
