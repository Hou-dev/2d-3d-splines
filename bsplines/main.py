import numpy as np
import scipy.interpolate as si


def bspline(cv, n=100, degree=3):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(0,(count-degree),n)

    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import itertools


class BsplineND():
    def __init__(self, knots, degree=3, periodic=False):
        """
        :param knots: a list of the spline knots with ndim = len(knots)
        """
        self.ndim = len(knots)
        self.splines = []
        self.knots = knots
        self.degree = degree
        for idim, knots1d in enumerate(knots):
            nknots1d = len(knots1d)
            y_dummy = np.zeros(nknots1d)
            knots1d, coeffs, degree = si.splrep(knots1d, y_dummy, k=degree,
                                                per=periodic)
            self.splines.append((knots1d, coeffs, degree))
        self.ncoeffs = [len(coeffs) for knots, coeffs, degree in self.splines]

    def evaluate(self, position):
        """
        :param position: a numpy array with size [ndim, npoints]
        :returns: a numpy array with size [nspl1, nspl2, ..., nsplN, npts]
                  with the spline basis evaluated at the input points
        """
        ndim, npts = position.shape

        values_shape = self.ncoeffs + [npts]
        values = np.empty(values_shape)
        ranges = [range(icoeffs) for icoeffs in self.ncoeffs]
        for icoeffs in itertools.product(*ranges):
            values_dim = np.empty((ndim, npts))
            for idim, icoeff in enumerate(icoeffs):
                coeffs = [1.0 if ispl == icoeff else 0.0 for ispl in
                          range(self.ncoeffs[idim])]
                values_dim[idim] = si.splev(
                        position[idim],
                        (self.splines[idim][0], coeffs, self.degree))

            values[icoeffs] = np.product(values_dim, axis=0)
        return values


def main():
    nx, ny = 11, 6
    nptsx, nptsy = 160, 80
    knotsx = np.arange(nx)
    knotsy = np.arange(ny)
    knots = [knotsx, knotsy]

    pointsx1d = np.linspace(knotsx[0], knotsx[-1], nptsx)
    pointsy1d = np.linspace(knotsy[0], knotsy[-1], nptsy)
    extent = (pointsx1d[0], pointsx1d[-1], pointsy1d[0], pointsy1d[-1])
    plotx, ploty = np.meshgrid(pointsx1d, pointsy1d, indexing='ij')
    points2d = np.array([plotx.flatten(), ploty.flatten()])

    periodic = False
    bspline1d = BsplineND([knotsx], periodic=periodic)
    values1d = bspline1d.evaluate(pointsx1d[None, :])

    bspline2d = BsplineND(knots, periodic=periodic)
    values2d = bspline2d.evaluate(points2d)

    # start plotting
    fig, ax = plt.subplots()
    for vals in values1d:
        ax.plot(pointsx1d, vals)
    fig.suptitle('1D Bspline basis from scipy (non-periodic)')

    if periodic:
        nsplx = nx + 2
        nsply = ny + 2
    else:
        nsplx = nx
        nsply = ny

    fig, axes = plt.subplots(nsply, nsplx, figsize=(0.8*nsplx, 0.5*nsply),
                             sharex=True, sharey=True)
    plt.setp(axes.flat, adjustable='box-forced')
    for icol in range(nsplx):
        for irow in range(nsply):
            ax = axes[irow, icol]
            ax.imshow(values2d[icol, irow].reshape(nptsx, nptsy).T,
                      extent=extent)
            ax.set(xlim=(knotsx[0], knotsx[-1]), ylim=(knotsy[0], knotsy[-1]))
    fig.suptitle('2D Bspline basis from scipy (non-periodic)')
    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(top=0.9)

    plt.show()


if __name__ == "__main__":
    main()