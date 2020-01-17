import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as my

# Imports for CSV file and reading into different arrays.
csv = np.genfromtxt(r'smooth_path.csv', delimiter=",", dtype=float)
x_axis = csv[:, 0]
y_axis = csv[:, 1]
z_axis = csv[:, 2]

# Import radius from CSV
csv2 = np.genfromtxt(r'GCorridor.csv', delimiter=",", dtype=float)
radius_data = csv[:, 1]


# Algorithm to find the control points

def de_castelijau(coord, t, a):
    p1 = ((1 - t) ** 3) * coord[a]
    p2 = 3 * t * ((1 - t) ** 2) * coord[a + 1]
    p3 = (3 * t ** 2) * (1 - t) * coord[a + 2]
    p4 = (t ** 3) * coord[a + 3]

    p1p = (p1 + p2) / 2
    p2p = (p2 + p3) / 2
    p3p = (p3 + p4) / 2

    p1pp = (p1p + p2p) / 2
    p2pp = (p2p + p3p) / 2

    p1ppp = (p1pp + p2pp) / 2

    return p1ppp


arr_count = 0
control_vector_x = []
control_vector_y = []
control_vector_z = []
for i in range(1736):
    control_vector_x.append(de_castelijau(x_axis, 0.2, i))
    control_vector_y.append(de_castelijau(y_axis, 0.2, i))
    control_vector_z.append(de_castelijau(z_axis, 0.2, i))
    arr_count = arr_count + 1
    i = i + 4

cv = np.empty((1730, 3))
for j in range(1730):
    cv[j] = control_vector_x[j], control_vector_y[j], control_vector_z[j]


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1)
    else:
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


p = bspline(cv, periodic=False)
x, y, z = p.T

# surface data
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.plot(ax, xs=x * 4, ys=y * 4, zs=z * 4, color='red')
Axes3D.plot(ax, xs=x_axis, ys=y_axis, zs=z_axis, color='blue')
plt.show()'''
s = radius_data/2
t = my.plot3d(x_axis, y_axis, z_axis, s, tube_radius=0.1)
t.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'
my.show()