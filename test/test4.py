"""


Author: Tong
Time: --2021
"""
import torch
from torch.nn.modules.rnn import LSTM
import numpy as np
import matplotlib.pyplot as plt

# create data points
x = np.linspace(-10, 10, 100)
y = np.linspace(-15, 15, 100)
# create grid
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)
# create figure container
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection='3d')
# 3d contour plot
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
# set labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# set point of view, altitude and azimuthal axis
ax.view_init(60, 95)
# save figure
plt.savefig('3d_surface.png', dpi=300)
