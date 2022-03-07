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
import math
from scipy.integrate import odeint

L1, L2 = 0.091, 0.07
m1, m2 = 0.01, 0.01
g = 9.81
R = 0.019
mu = 1e-5
c1 = 6*math.pi*mu*R
c2 = c1

parser = argparse.ArgumentParser()

parser.add_argument('-datasource', default='datasource_2', type=str)
parser.add_argument('-batch_size', default=1, type=int)
parser.add_argument('-sequence_len', default=100, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-train_size', default=0.8, type=int)
parser.add_argument('-num_workers', default=0, type=int)
parser.add_argument('-output_directory', default='D:/bakalaura_darbs/v3', type=str)
parser.add_argument('-frame_count', default=4, type=int)

args, args_other = parser.parse_known_args()

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

def initial_angular_speed(angles_1, angles_2,starting_idx, frame_count, frequency_hertz):
    angle_initial_1 = angles_1[starting_idx]
    angle_end_1 = angles_1[starting_idx + frame_count]

    angle_initial_2 = angles_2[starting_idx]
    angle_end_2 = angles_2[starting_idx + frame_count]

    delta_1 = (angle_end_1 - angle_initial_1)
    delta_2 = (angle_end_2 - angle_initial_2)

    if delta_1 > 2: #check for jump from 2pi to 0
        delta_1 = (angle_end_1 - (angle_initial_1-2*math.pi))
    if delta_2 > 2:
        delta_1 = (angle_end_2 - (angle_initial_2-2*math.pi))

    time = frame_count * 1/frequency_hertz
    speed_1 = delta_1/time
    speed_2 = delta_2/time
    return speed_1, speed_2

def deriv(y,t, L1, L2, m1, m2, c1,c2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = -(g * np.sin(theta1)*(m1+m2) \
            + L1**2*theta2*(c1+c2) \
            + L2*m2*s*z2**2 \
            - L2*c2*c*z2 \
            - g*m2*c*np.sin(theta2) \
            - L1*L2*c2*c**2*z1 \
            + L1*m2*c*s*z1**2 \
            + L1*L2*c2*c*z2) \
            / (L1*(m1 + m2) - L1*m2*c**2)

    theta2dot = z2
    z2dot = (g*m2**2*c*np.sin(theta1) \
            - g*m2**2*np.sin(theta2) \
            - L2*c2*z2*(m1+m2) \
            + L1*m2**2*s*z1**2 \
            - g*m1*m2*np.sin(theta2) \
            + L2*m2**2*c*s*z2**2 \
            + g*m1*m2*c*np.sin(theta1) \
            + L1**2*c1*m2*c*z1 \
            + L1**2*c2*m2*c*z1 \
            + L1*m1*m2*s*z1**2 \
            + L1*L2*c2*c* (-m1*z1 -m2*z1 + m2*z2 ) ) \
            / (L2*m2**2 - L2*m2**2*c**2 + L2*m1*m2)

    return theta1dot, z1dot, theta2dot, z2dot


get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']),
                           'get_data_loaders')
dataset_train, dataset_test, max_value, min_value = get_data_loaders(args)

t = np.array([0,0.0025])
loss_holder = []
for x, y in tqdm(dataset_test):
    x = x.squeeze().numpy()
    y = y.squeeze().numpy()
    working_x = x[args.frame_count:,:]
    working_y = y[args.frame_count:,:]
    for idx, angles in enumerate(working_x):
        omega1, omega2 = initial_angular_speed(x[:, 0], x[:, 1], starting_idx=idx, frame_count=args.frame_count, frequency_hertz=400)
        y0 = np.array([angles[0], omega1, angles[1], omega2])

        # Do the numerical integration of the equations of motion
        ode_next_step = odeint(deriv, y0, t, args=(L1, L2, m1, m2, c1, c2))

        ode_thetas_next = np.array([ode_next_step[1][0], ode_next_step[1][2]])
        loss = np.mean(np.abs(ode_thetas_next - working_y[idx]))
        loss_holder.append(loss)

    print(np.mean(np.array(loss_holder)))

loss_total_mean = np.mean(np.array(loss_holder))
