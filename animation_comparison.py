import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from csv import reader
import math

#avots piemēram https://scipython.com/blog/the-double-pendulum/

L1, L2 = 1, 1
m1, m2 = 1, 1
g = 9.81

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    """Return the total energy of the system."""

    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

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

theta1_l = []
theta2_l = []

for row in coordinates:
    angle = raw_cartesian_to_polar_angles(row)
    theta1_l.append(angle[0])
    theta2_l.append(angle[1])

# Maximum time, time point spacings and the time grid (all in s).
length = 30
hertz = 400
t = np.arange(0, length, 1/hertz)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
omega1_l, omega2_l  = initial_angular_speed(theta1_l, theta2_l, frame_count=4, frequency_hertz=400 )
y0 = np.array([theta1_l[0], omega1_l, theta2_l[0], omega2_l])

# Do the numerical integration of the equations of motion
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

theta1, theta2 = y[:,0], y[:,2]

# Convert to Cartesian coordinates of the two bob positions.
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

x1_l = L1 * np.sin(theta1_l)
y1_l = -L1 * np.cos(theta1_l)
x2_l = x1_l + L2 * np.sin(theta2_l)
y2_l = y1_l - L2 * np.cos(theta2_l)
#(x_red, y_red, x_green, y_green, x_blue, y_blue)
# Plotted bob circle radius
r = 0.05

def make_plot(i,ax,ode=True):
    if ode:
        t1, t2, t3, t4 = x1, x2, y1, y2
    else:
        t1, t2, t3, t4 = x1_l, x2_l, y1_l, y2_l

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

fig = plt.figure(figsize=(16, 10), dpi=72)
ax1 = plt.subplot(121)
ax1.set_title('Experimental')
ax2 = plt.subplot(122)
ax2.set_title('ODE')

for i in range(0, int(length * hertz), 1):
    # print(i // di, '/', t.size // di)
    make_plot(i, ax1, ode=False)
    ax1.set_title('Experimental', fontsize = 20)
    make_plot(i, ax2)
    ax2.set_title('ODE', fontsize = 20)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    ax1.clear()
    ax2.clear()
