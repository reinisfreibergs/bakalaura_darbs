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


FRAMES = 800 #how many frames the animation will include
START_FRAMES = 401  #how many frames to feed in
SEQUENCE_LENGTH = 400 #original sequence length
if START_FRAMES==0:
    START_FRAMES = SEQUENCE_LENGTH
START = 100

# model_path = f'./results/hpc_19-03-22/sequence-0-run-22-03-16--20-24-07/model_test_{SEQUENCE_LENGTH}_test_loss.pt' # 2048 hidden modelis
model_path = f'./results/v4_10/run-22-03-20--23-24-46/model_test_{SEQUENCE_LENGTH}_test_loss.pt' # 64 hidden modelis


parser = argparse.ArgumentParser()
# parser.add_argument('-model', default='model_1_LSTM', type=str)
# parser.add_argument('-model', default='model_2_phased_lstm', type=str)
# parser.add_argument('-model', default='model_3_PLSTM', type=str)
# parser.add_argument('-model', default='model_4_snake_LSTM', type=str)
# parser.add_argument('-model', default='model_5_hidden', type=str)
parser.add_argument('-model', default='model_6_hidden', type=str)

parser.add_argument('-hidden_size', default=64, type=int)
parser.add_argument('-sequence_len', default=400, type=int)
parser.add_argument('-device', default='cpu', type=str)
parser.add_argument('-lstm_layers', default=10, type=int)
parser.add_argument('-batch_size', default=128, type=int)

parser.add_argument('-activation', default='mish', type=str)
parser.add_argument('-maxout_layers', default=2, type=int)
parser.add_argument('-is_zero_test', default=False, type=lambda x: (str(x).lower() == 'true'))


args = parser.parse_args()

def param_count(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
    return total_param_size

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
print(param_count(model))

with open('../data/original/dpc_dataset_csv/20.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    coordinates = list(csv_reader)

start_angles = [] #starting coordinates for model
for row_idx in range(START, START+START_FRAMES+FRAMES):
    angle = coordinates[row_idx]
    angle = raw_cartesian_to_polar_angles(angle)
    start_angles.append(angle)

start_angles = np.array(start_angles)
angles_next = np.roll(start_angles, shift=1, axis=0)
initial_diff = angles_next - start_angles
initial_diff[initial_diff<-4] = angles_next[initial_diff<-4] - (start_angles[initial_diff<-4] - 2*math.pi)
initial_diff[initial_diff>4] = (angles_next[initial_diff>4] - 2*math.pi) - start_angles[initial_diff>4]

usable_diff = initial_diff[1:START_FRAMES]
if args.is_zero_test:
    usable_diff = torch.zeros_like(torch.FloatTensor(usable_diff))
    initial_diff = torch.zeros_like(torch.FloatTensor(initial_diff))
usable_angles_next = angles_next[1:START_FRAMES]

start = torch.reshape(torch.FloatTensor(usable_diff), shape=(1, SEQUENCE_LENGTH, 2)).to(args.device)
start_unused = start
initial_angle = usable_angles_next[-1]


diffs = torch.FloatTensor().to(args.device)

fig, ax = plt.subplots()
camera = Camera(fig)
state = None

with torch.no_grad():
    for i in tqdm(range(FRAMES)):
        diffs_current, state = model.forward(start, state)

        # plt.plot(400*diffs_current[0,:,0], label='$\omega_1$ modeļa prognoze')
        # plt.plot(400*initial_diff[2:START_FRAMES+1][:,0], label='$\omega_1$ patiesā vērtība')
        # plt.legend()
        # plt.title('1 soļa prognoze')
        # plt.xlabel('Laika solis t, 0.025s ')
        # plt.ylabel('Leņķiskais ātrums $\omega$, rad/s')
        # plt.show()

        diffs = torch.cat(tensors=(diffs, diffs_current[:,-1,:].unsqueeze(dim=0)), dim=1)

        # plt.plot(diffs[0,:,0])
        # # ax.plot(angles.cpu()[0,:,0])
        # ax.plot(start[0,:,0])
        # plt.draw()
        # plt.pause(0.1)
        # plt.cla()

        start = diffs_current[:,-1,:].unsqueeze(dim=1)

        # start_1 = torch.cat(tensors=(start[:,1:,:],diffs_current[:,-1,:].unsqueeze(dim=0)), dim=1 )

        # plt.plot(diffs.squeeze())
        # plt.draw()
        # plt.pause(0.1)
        # plt.cla()

        # start = diffs_current #iespēja modelim dot visu garo sakrāto sequence vai katrā solī sākumā noteikto sequence garumu.
        # start = diffs


diff1 = diffs[:,:,0].cpu().squeeze().detach().numpy()
diff2 = diffs[:,:,1].cpu().squeeze().detach().numpy()
plt.plot(diff1)
plt.plot(diff2)
plt.show()

plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH+len(diff1), 1), 400*diff1, color='b', label='$\omega_1$ modeļa prognoze')
plt.plot(400*initial_diff[1:START_FRAMES, 0], color='g', label='$\omega_1$ modelī padotā vērtība')
plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH+len(diff1), 1), 400*initial_diff[START_FRAMES:, 0], color='r', label='$\omega_1$ patiesā vērtība')
plt.title('$\omega_1$ teorētiskie un prognozētie leņķiskie ātrumi')
plt.xlabel('laika solis t, 0.025s ')
plt.ylabel('leņķiskais ātrums $\omega$, rad/s')
plt.legend()
plt.show()

plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH+len(diff2), 1), 400*diff2, color='b', label='$\omega_2$ modeļa prognoze')
plt.plot(400*initial_diff[1:START_FRAMES, 1], color='g', label='$\omega_2$ modelī padotā vērtība')
plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH+len(diff2), 1), 400*initial_diff[START_FRAMES:, 1], color='r', label='$\omega_2$ patiesā vērtība')
plt.title('$\omega_2$ teorētiskie un prognozētie leņķiskie ātrumi')
plt.xlabel('Laika solis t, 0.025s ')
plt.ylabel('Leņķiskais ātrums $\omega$, rad/s')
plt.legend()
plt.show()

# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot(usable_diff)
# ax2.plot(diff1)
# ax2.plot(diff2)
# fig.suptitle('Leņķisko ātrumu salīdzinājums')
# ax2.legend('omega 1', 'omega 2')
# ax2.set_xlabel('laika solis')
# ax2.set_ylabel('leņķiskais ātrums, rad')
# plt.show()

angle_0 = initial_angle[0]
theta_1 = []
for diff in diff1:
    angle = angle_0 - diff
    theta_1.append(angle)
    angle_0 = angle
theta_1 = np.array(theta_1)

angle_0 = initial_angle[1]
theta_2 = []
for diff in diff2:
    angle = angle_0 - diff
    theta_2.append(angle)
    angle_0 = angle
theta_2 = np.array(theta_2)


L1, L2 = 0.091, 0.07
m1, m2 = 0.01, 0.01
g = 9.81
r = 0.005

x1 = L1 * np.sin(theta_1)
y1 = -L1 * np.cos(theta_1)
x2 = x1 + L2 * np.sin(theta_2)
y2 = y1 - L2 * np.cos(theta_2)


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
