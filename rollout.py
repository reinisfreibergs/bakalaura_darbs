import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
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
import time

HIDDEN_SIZE = 64

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=HIDDEN_SIZE),
            torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        )
        self.lstm_layer = torch.nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True, num_layers=2)
        self.linear_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE),
            torch.nn.Mish(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=4)
        )
    def forward(self, x):
        y_1 = self.linear_1.forward(x)
        lstm_out, _ = self.lstm_layer.forward(y_1)
        y_prim = self.linear_2.forward(lstm_out)

        return y_prim

model_path = './results/model_test_1.pt'
model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()

start_angle_1 = 45
start_angle_2 = 15
angle_1 = 44.98
angle_2 = 14.98

start_sin_cos = [[math.sin(math.radians(start_angle_1)),
                 math.cos(math.radians(start_angle_1)),
                 math.sin(math.radians(start_angle_2)),
                 math.cos(math.radians(start_angle_2))],
                 [math.sin(math.radians(angle_1)),
                 math.cos(math.radians(angle_1)),
                 math.sin(math.radians(angle_2)),
                 math.cos(math.radians(angle_2))]]

start = torch.reshape(torch.FloatTensor(start_sin_cos), shape=(1,2,4))

angles = torch.FloatTensor()
# angles += start
with torch.no_grad():
    for i in range(5):
        angles_current = model.forward(start)
        angles = torch.cat(tensors=(angles, angles_current), dim=1)
        start = angles_current


sin_theta1 = angles[:,:,0].squeeze().detach().numpy()
cos_theta1 = angles[:,:,1].squeeze().detach().numpy()
sin_theta2 = angles[:,:,2].squeeze().detach().numpy()
cos_theta2 = angles[:,:,3].squeeze().detach().numpy()

theta_1 = np.rad2deg(np.arctan2(sin_theta1, cos_theta1))
theta_2 = np.rad2deg(np.arctan2(sin_theta2, cos_theta2))

L1, L2 = 0.091, 0.07
m1, m2 = 0.01, 0.01
g = 9.81
r = 0.005

x1 = L1 * sin_theta1
y1 = -L1 * cos_theta1
x2 = x1 + L2 * sin_theta2
y2 = y1 - L2 * cos_theta2


def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

fig, ax = plt.subplots(1,1, figsize=(8,8))
ln1, = plt.plot([], [], lw=2, c='k', markersize=8)

ax.set_xlim(-L1-L2-r, L1+L2+r)
ax.set_ylim(-L1-L2-r, L1+L2+r)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

ani = animation.FuncAnimation(fig, animate, frames=800, interval=1000/400, repeat=False)
ani.save('rollout_3.mp4', fps=130) #150??? 1000/400 = 2.5 ms per frame -> 400 fps
# plt.show()
