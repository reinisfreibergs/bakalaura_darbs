import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
import numpy as np
import math
from csv import reader

HIDDEN_SIZE = 64
FRAMES = 800
SEQUENCE_LENGTH = 100
START = 0

def raw_cartesian_to_polar_angles(l):
    '''Convert the cartesian coordinates to polar coordinates.'''
    x_red, y_red, x_green, y_green, x_blue, y_blue = [int(x) for x in l]

    angle_green_red = math.atan2((y_green-y_red),(x_green-x_red))
    angle_blue_green = math.atan2((y_blue-y_green),(x_blue-x_green))

    return [np.sin(angle_green_red), np.cos(angle_green_red), np.sin(angle_blue_green), np.cos(angle_blue_green)]

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

model_path = './results/model_test_3_single_epoch.pt'
model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()

with open('0.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    coordinates = list(csv_reader)

start_sin_cos = []
for row_idx in range(START, START+SEQUENCE_LENGTH):
    angle = coordinates[row_idx]
    angle = raw_cartesian_to_polar_angles(angle)
    start_sin_cos.append(angle)

start = torch.reshape(torch.FloatTensor(start_sin_cos), shape=(1,SEQUENCE_LENGTH,4))

angles = torch.FloatTensor()
angles = torch.cat(tensors=(angles, start), dim=1)

with torch.no_grad():
    for i in range(FRAMES):
        angles_current = model.forward(start)
        angles = torch.cat(tensors=(angles, angles_current[:,-1,:].unsqueeze(dim=0)), dim=1)
        # angles = torch.cat(tensors=(angles, angles))
        # start = angles_current
        start = angles


sin_theta1 = angles[:,:,0].squeeze().detach().numpy()
cos_theta1 = angles[:,:,1].squeeze().detach().numpy()
sin_theta2 = angles[:,:,2].squeeze().detach().numpy()
cos_theta2 = angles[:,:,3].squeeze().detach().numpy()

theta1 = np.arctan2(sin_theta1, cos_theta1)
theta2 = np.arctan2(sin_theta2, cos_theta2)

L1, L2 = 0.091, 0.07
m1, m2 = 0.01, 0.01
g = 9.81
r = 0.005

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)


def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

fig, ax = plt.subplots(1,1, figsize=(8,8))
ln1, = plt.plot([], [], lw=2, c='k', markersize=8)

ax.set_xlim(-L1-L2-r, L1+L2+r)
ax.set_ylim(-L1-L2-r, L1+L2+r)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

ani = animation.FuncAnimation(fig, animate, frames=FRAMES, interval=1000/400, repeat=False)
ani.save('rollout_single_epoch_test_100_2.mp4', fps=130) #150??? 1000/400 = 2.5 ms per frame -> 400 fps
# plt.show()
