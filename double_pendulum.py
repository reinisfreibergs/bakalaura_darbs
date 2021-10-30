import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import math

train_file = '0.csv'

def raw_to_pixel(l):
    '''Convert the raw coordinates to pixel coordinates.'''
    return [int(x)/5 for x in l]

def raw_cartesian_to_polar_angles(l):
    '''Convert the cartesian coordinates to polar coordinates.'''
    x_red, y_red, x_green, y_green, x_blue, y_blue = raw_to_pixel(l)

    angle_green_red = np.arctan((y_green-y_red)/(x_green-x_red+1e-5))
    angle_blue_green = np.arctan((y_blue-y_green)/(x_blue-x_green+1e-5))

    return [np.degrees(angle_green_red), np.degrees(angle_blue_green)]


def initial_angular_speed(angles_1, angles_2, frame_count, frequency_hertz):
    angle_initial_1 = angles_1[0]
    angle_end_1 = angles_1[frame_count]

    angle_initial_2 = angles_2[0]
    angle_end_2 = angles_2[frame_count]

    delta_1 = math.pi/180*(angle_end_1 - angle_initial_1)
    delta_2 = math.pi/180*(angle_end_2 - angle_initial_2)

    time = frame_count * 1/frequency_hertz
    speed_1 = delta_1/time
    speed_2 = delta_2/time
    return speed_1, speed_2

with open('0.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    coordinates = list(csv_reader)

#(x_red, y_red, x_green, y_green, x_blue, y_blue)
angles_1 = []
angles_2 = []
coordinates = coordinates[15000:]

for row in coordinates:
    angle = raw_cartesian_to_polar_angles(row)
    angles_1.append(angle[0])
    angles_2.append(angle[1])

print(initial_angular_speed(angles_1, angles_2, 40, 400))
exit()
t = np.linspace(0,len(coordinates)/400, len(coordinates))
plt.figure(figsize=(10,10))
plt.plot(t, angles_2)
# plt.xlim([0,40])
plt.show()



