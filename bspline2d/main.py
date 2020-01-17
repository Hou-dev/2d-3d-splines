import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

# Actual Data to compare to B Spline
num_sample_pts = 100
x_real = np.arange(0, num_sample_pts, 0.1, float)
y_real = np.sin(x_real)

# Finds the B Spline, using parameters x,y and k which is the degree
# tck is the return of vector knots, B Spline coefficients and degree
# splev will return values of the spline function at x
# using 200 data points, we return a curve vs 1000 points in actual
data_points = 200
num_sample_spline = 100
tck = spi.splrep(x_real, y_real, k=3)
x_interp = np.linspace(0, num_sample_spline, data_points)
y_interp = spi.splev(x_interp, tck)

'''plt.plot(x_interp, y_interp, 'r')
plt.plot(x_real, y_real, 'b')'''


# Plotting Data
fig, ax = plt.subplots()
ax.plot(x_real, y_real, label='Actual')
ax.plot(x_interp, y_interp, label='B Spline')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.grid(True)
plt.title('Plot of Cubic B-Spline Interpolation vs Actual')
plt.legend()
plt.show()

'''
t, c, k = spi.splrep(x_real, y_real, s=0, k=3)
x_min, x_max = x_real.min(), x_real.max()
x_interp = np.linspace(x_min, x_max, 100)
y_interp = spi.BSpline(t, c, k, extrapolate=False)


plt.plot(x_real, y_real, 'b')
plt.plot(x_interp, y_interp(x_interp), 'r')
plt.show()
'''