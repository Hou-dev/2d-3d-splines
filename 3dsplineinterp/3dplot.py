import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Imports for CSV file and reading into different arrays.
csv = np.genfromtxt(r'smooth_path.csv', delimiter=",", dtype=float)
x_axis = csv[:, 0]
y_axis = csv[:, 1]
z_axis = csv[:, 2]


# Import radius from CSV
csv2 = np.genfromtxt(r'GCorridor.csv', delimiter=",", dtype=float)
radius_data = csv[:, 1]

# Takes axis and reads into bisplev, which does the representation on a surface
tck = spi.bisplrep(x_axis, y_axis, z_axis)
x_interpolate = csv[:, 0]
y_interpolate = csv[:, 1]

# bisplev does the evaluation on the z axis
# z_interpolate = spi.bisplev(x_interpolate, y_interpolate, tck)

# surface data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.plot(ax, xs=x_axis, ys=y_axis, zs=z_axis, color='red')
plt.show()


