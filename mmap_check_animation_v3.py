import json
import random

import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
from tqdm import tqdm
from modules_core.datasource_2 import Dataset_time_series
from matplotlib.patches import Circle
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument('-datasource', default='datasource_2', type=str)

parser.add_argument('-sequence_len', default=1000, type=int)
parser.add_argument('-csv_directory', default='./dummy_csv_2', type=str) #'../data/original/dpc_dataset_csv', './dummy_csv'
parser.add_argument('-output_directory', default='./datasource_dummy', type=str)
parser.add_argument('-is_overfitting_test', default=False, type=lambda x: (str(x).lower() == 'false'))
parser.add_argument('-train_size', default=0.8, type=int)
parser.add_argument('-num_workers', default=0, type=int)
args = parser.parse_args()

L1, L2 = 0.091, 0.07
m1, m2 = 0.01, 0.01
g = 9.81
r = 0.005

def make_plot(i,ax, t1, t2, t3, t4):
    ax.plot([0, t1[i], t2[i]], [0, t3[i], t4[i]], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((t1[i], t3[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((t2[i], t4[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

# get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']),
#                            'get_data_loaders')
# dataset_train, dataset_test, max_value, min_value = get_data_loaders(args)
# init_dataset = dataset_train.dataset[-10:]
# # k = init_dataset[-1000] #random sample
# theta1, theta2 = init_dataset
init_dataset = Dataset_time_series(args)
k = init_dataset[0] #random sample
theta1, theta2 = k[0][:,0], k[0][:,1]
# sin_theta1, cos_theta1, sin_theta2, cos_theta2 = k[1][:,0], k[1][:,1], k[1][:,2], k[1][:,3]
# k = init_dataset.data[-1000:,]
p=1
# theta1 = np.arctan2(sin_theta1, cos_theta1)
# theta2 = np.arctan2(sin_theta2, cos_theta2)

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)


# fig = plt.figure(figsize=(16, 10), dpi=72)
# ax = plt.subplot()
#
# for i in range(0, 4000, 1):
#     # print(i // di, '/', t.size // di)
#     make_plot(i,ax, x1, x2, y1, y2)
#     ax.set_title('Experimental', fontsize = 20)
#     plt.draw()
#     plt.pause(0.1)
#     ax.clear()


def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

fig, ax = plt.subplots(1,1, figsize=(8,8))
ln1, = plt.plot([], [], lw=2, c='k', markersize=8)

ax.set_xlim(-L1-L2-r, L1+L2+r)
ax.set_ylim(-L1-L2-r, L1+L2+r)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

# Writer = animation.writers['ffmpeg']
# writer1 = Writer(fps = 400)
ani = animation.FuncAnimation(fig, animate, frames=800, interval=1000/400, repeat=False)
# ani = animation.FuncAnimation(fig, animate, frames=800, repeat=False)
ani.save('test_v3.mp4', fps=150) #150??? 1000/400 = 2.5 ms per frame -> 400 fps
plt.show()
