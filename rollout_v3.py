import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
import numpy as np
import math
from csv import reader
from tqdm import tqdm
from datetime import datetime
import matplotlib.animation as animation
from celluloid import Camera
import argparse

# HIDDEN_SIZE = 64
FRAMES = 1600 #how many frames the animation will include
START_FRAMES = 400  #how many frames to feed in
SEQUENCE_LENGTH = 400 #original sequence length
if START_FRAMES==0:
    START_FRAMES = SEQUENCE_LENGTH
START = 0

model_path = f'./results/v3_test/run-22-02-27--10-46-17/model_test_{SEQUENCE_LENGTH}_test_loss.pt'
# model_path = f'./results/usable_snake/run-22-02-21--03-04-05/model_test_{SEQUENCE_LENGTH}_test_loss.pt'

parser = argparse.ArgumentParser()
# parser.add_argument('-model', default='model_1_LSTM', type=str)
# parser.add_argument('-model', default='model_2_phased_lstm', type=str)
# parser.add_argument('-model', default='model_3_PLSTM', type=str)
# parser.add_argument('-model', default='model_4_snake_LSTM', type=str)
parser.add_argument('-model', default='model_5_hidden', type=str)

parser.add_argument('-hidden_size', default=16, type=int)
parser.add_argument('-sequence_len', default=400, type=int)
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-lstm_layers', default=2, type=int)

parser.add_argument('-activation', default='snake', type=str)
parser.add_argument('-maxout_layers', default=2, type=int)


args = parser.parse_args()

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


Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
model = Model(args).to(args.device)
model.load_state_dict(torch.load(model_path))
model = model.eval()
torch.set_grad_enabled(False)

with open('0.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    coordinates = list(csv_reader)

start_sin_cos = [] #starting coordinates for model
for row_idx in range(START, START+START_FRAMES):
    angle = coordinates[row_idx]
    angle = raw_cartesian_to_polar_angles(angle)
    start_sin_cos.append(angle)

check_sin_cos = [] #coordinates that should be predicted
for row_idx in range(START+START_FRAMES, (START+START_FRAMES) + FRAMES):
    angle = coordinates[row_idx]
    angle = raw_cartesian_to_polar_angles(angle)
    check_sin_cos.append(angle)

start = torch.reshape(torch.FloatTensor(start_sin_cos), shape=(1,START_FRAMES,2)).to(args.device)
#
# fig, axs = plt.subplots(4,1)
# axs[0].plot(start[0,:,0])
# axs[1].plot(start[0,:,1])
# axs[2].plot(start[0,:,2])
# axs[3].plot(start[0,:,3])
# plt.show()
# exit()

# theta1 = np.arctan2(start[0,:,0], start[0,:,1])
# theta2 = np.arctan2(start[0,:,2], start[0,:,3])
# fig, axs = plt.subplots(2,1)
# axs[0].plot(theta1)
# axs[1].plot(theta2)
# plt.show()
# exit()


angles = torch.FloatTensor().to(args.device)

fig, ax = plt.subplots()
camera = Camera(fig)
state = None
with torch.no_grad():
    for i in tqdm(range(FRAMES)):
        angles_current, state = model.forward(start, state)
        # angles_current = angles_current.unsqueeze(dim=0)
        # plt.plot(angles_current[0,:,0])
        # plt.show()

        angles = torch.cat(tensors=(angles, angles_current[:,-1,:].unsqueeze(dim=0)), dim=1)
        # ax.plot(angles.cpu()[0,:,0])
        # # # ax.plot(angles_current[0,:,0])
        # plt.draw()
        # plt.pause(0.1)
        # plt.cla()
        start = angles_current[:,-1,:].unsqueeze(dim=1)
        # start = torch.cat(tensors=(start[:,1:,:],angles_current[:,-1,:].unsqueeze(dim=0)), dim=1 )
        # start = angles_current #iespēja modelim dot visu garo sakrāto sequence vai katrā solī sākumā noteikto sequence garumu.
        # start = angles

def animate(i):
    # return angles[0,i,0]
    ln1.set_ydata(angles[0,i,0].item())

# fig1, ax1 = plt.subplots()
#
# ln1, = plt.plot([])
# ani = animation.FuncAnimation(fig1, animate, frames=FRAMES, interval=100, repeat=True)
# animation  = camera.animate()

# plt.show()
# exit()

theta1 = angles[:,:,0].cpu().squeeze().detach().numpy()
theta2 = angles[:,:,1].cpu().squeeze().detach().numpy()


# fig, axs = plt.subplots(4,1)
# axs[0].plot(sin_theta1)
# axs[1].plot(cos_theta1)
# axs[2].plot(sin_theta2)
# axs[3].plot(cos_theta2)
# plt.show()
# exit()
# fig, axs = plt.subplots(2,1)
# axs[0].plot(theta1)
# axs[1].plot(theta2)
# plt.show()
# exit()

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

ani = animation.FuncAnimation(fig, animate, frames=FRAMES, interval=1000/400, repeat=True)
# ani.save(f'rollout/rollout_{FRAMES/400}s_seq_len{SEQUENCE_LENGTH}_{datetime.utcnow().strftime(f"%y-%m-%d--%H-%M-%S")}.mp4', fps=130) #150??? 1000/400 = 2.5 ms per frame -> 400 fps
plt.show()
