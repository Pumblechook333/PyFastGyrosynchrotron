import math
import numpy as np
from numba.experimental import jitclass
from numba import njit #, jit
import numba as nb

# USED TO INSTANCE ALL NUMERICAL POINTERS
bigNeg = -9.2559631349317831e+61
dNaN = np.inf  # will be changed to huge double val
JMAX = 20


@njit
def spline_init(x, y, n, yp1, ypn, y2):
    """Performs the initial piecewise polynomial interpolation calculations for a Spline object, loading the
       extrapolated data points into the y2 array."""

    u = np.full(n, bigNeg)

    if not (np.isfinite(yp1)):
        y2[0] = 0.0
        u[0] = 0.0
    else:
        y2[0] = -0.5
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)

    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    if not (np.isfinite(ypn)):
        qn = 0.0
        un = 0.0
    else:
        qn = 0.5
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))

    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)

    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]


@njit
def spline_short(x, y, n, y2):
    """An intermediary function which quickly prepares the initial parameters of a Spline object for further
       interpolation.

       Calls the spline_init function."""

    K1 = np.full(3, bigNeg)

    dxl = x[1] - x[0]
    dxr = x[2] - x[1]
    K1[0] = -(2.0 * dxl + dxr) / dxl / (dxl + dxr)
    K1[1] = (dxr + dxl) / (dxr * dxl)
    K1[2] = -dxl / dxr / (dxr + dxl)

    y1l = K1[0] * y[0] + K1[1] * y[1] + K1[2] * y[2]

    dxl = x[n - 2] - x[n - 3]
    dxr = x[n - 1] - x[n - 2]
    K1[0] = dxr / dxl / (dxr + dxl)
    K1[1] = -(dxr + dxl) / (dxr * dxl)
    K1[2] = (2.0 * dxr + dxl) / dxr / (dxr + dxl)

    y1r = K1[0] * y[n - 3] + K1[1] * y[n - 2] + K1[2] * y[n - 1]

    spline_init(x, y, n, y1l, y1r, y2)


@njit
def spline_interp(xa, ya, y2a, n, x, y, y1):
    """Performs the full polynomial interpolation calculation on an existing Spline object, callable on an already
       prepared set of data points."""

    if x <= xa[0]:
        klo = 0
        khi = 1
    elif x >= xa[n - 1]:
        klo = n - 2
        khi = n - 1
    else:
        klo = 0
        khi = n - 1
        while (khi - klo) >> 1:
            k = (khi + klo) >> 1
            if xa[k] > x:
                khi = k
            else:
                klo = k

    h = xa[khi] - xa[klo]
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h

    if y:
        y[0] = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0
    if y1:
        y1[0] = (ya[khi] - ya[klo]) / h + ((1.0 - 3.0 * a * a) * y2a[klo] + (3.0 * b * b - 1) * y2a[khi]) * h / 6.0


specSpline = [
    ('N', nb.int64),
    ('x_arr', nb.float64[:]),
    ('y_arr', nb.float64[:]),
    ('y2_arr', nb.float64[:])
]


@jitclass(specSpline)
class Spline:
    """An object meant to handle the necessary calculations for a polynomial-based interpolation (estimation) method."""

    def __init__(self, _N=None, x=None, y=None):

        self.N = _N
        self.x_arr = np.full(self.N, bigNeg)
        self.y_arr = np.full(self.N, bigNeg)
        self.y2_arr = np.full(self.N, bigNeg)

        self.x_arr = x
        self.y_arr = y

        spline_short(self.x_arr, self.y_arr, self.N, self.y2_arr)

    def Interpolate(self, x, y, y1):
        """Calls the interpolation of data from new data x and y, inserting the calculated data into y1."""

        spline_interp(self.x_arr, self.y_arr, self.y2_arr, self.N, x, y, y1)


specSpline2D = [
    ('Nx', nb.int64),
    ('Ny', nb.int64),

    ('x_arr', nb.float64[:]),
    ('y_arr', nb.float64[:]),

    ('f_arr', nb.float64[:, :]),
    ('f2_xx_arr', nb.float64[:, :]),
    ('f2_yy_arr', nb.float64[:, :]),
    ('f4_xxyy_arr', nb.float64[:, :]),
]


@jitclass(specSpline2D)
class Spline2D:
    """A modified version of the Spline object handler that is capable of interpolation of two-dimensional arrays."""

    def __init__(self, N_x=None, N_y=None, x=None, y=None, f=None):

        self.Nx = N_x
        self.Ny = N_y

        self.x_arr = np.full(self.Nx, bigNeg)
        self.y_arr = np.full(self.Ny, bigNeg)

        self.f_arr = np.full((self.Nx, self.Ny), bigNeg)
        self.f2_xx_arr = np.full((self.Nx, self.Ny), bigNeg)
        self.f2_yy_arr = np.full((self.Nx, self.Ny), bigNeg)
        self.f4_xxyy_arr = np.full((self.Nx, self.Ny), bigNeg)

        self.x_arr = x
        self.y_arr = y
        self.f_arr = f

        xprof2 = np.full(self.Nx, bigNeg)

        for j in range(0, self.Ny):
            xprof = self.f_arr[:, j]
            spline_short(x, xprof, self.Nx, xprof2)
            self.f2_xx_arr[:, j] = xprof2

        for i in range(0, self.Nx):
            spline_short(y, self.f_arr[i], self.Ny, self.f2_yy_arr[i])
            spline_short(y, self.f2_xx_arr[i], self.Ny, self.f4_xxyy_arr[i])

    def Interpolate(self, x, y, f, f_x, f_y, f2_yy):
        """interpolates data x and y using f, loading the calculated data into f_x, f_y and f2_yy"""

        if x <= self.x_arr[0]:
            i1 = 0
            i2 = 1
        elif x >= self.x_arr[self.Nx - 1]:
            i1 = self.Nx - 2
            i2 = self.Nx - 1
        else:
            i1 = 0
            i2 = self.Nx - 1
            while i2 - i1 > 1:
                k = (i1 + i2) >> 1
                if self.x_arr[k] > x:
                    i2 = k
                else:
                    i1 = k

        hx = self.x_arr[i2] - self.x_arr[i1]
        ax = (self.x_arr[i2] - x) / hx
        bx = (x - self.x_arr[i1]) / hx

        if y <= self.y_arr[0]:
            j1 = 0
            j2 = 1
        elif y >= self.y_arr[self.Ny - 1]:
            j1 = self.Ny - 2
            j2 = self.Ny - 1
        else:
            j1 = 0
            j2 = self.Ny - 1
            while j2 - j1 > 1:
                k = (j1 + j2) >> 1
                if self.y_arr[k] > y:
                    j2 = k
                else:
                    j1 = k

        hy = self.y_arr[j2] - self.y_arr[j1]
        ay = (self.y_arr[j2] - y) / hy
        by = (y - self.y_arr[j1]) / hy

        f_lo = ay * self.f_arr[i1, j1] + by * self.f_arr[i1, j2] + ((ay * ay * ay - ay) * self.f2_yy_arr[i1, j1] + (
               by * by * by - by) * self.f2_yy_arr[i1, j2]) * (hy * hy) / 6.0
        f_y_lo = (self.f_arr[i1, j2] - self.f_arr[i1, j1]) / hy + ((1.0 - 3.0 * ay * ay) * self.f2_yy_arr[i1, j1] + (
                 3.0 * by * by - 1) * self.f2_yy_arr[i1, j2]) * hy / 6.0

        f2_yy_lo = ay * self.f2_yy_arr[i1, j1] + by * self.f2_yy_arr[i1, j2]
        f2_xx_lo = ay * self.f2_xx_arr[i1, j1] + by * self.f2_xx_arr[i1, j2] + ((ay * ay * ay - ay) * self.f4_xxyy_arr[
                   i1, j1] + (by * by * by - by) * self.f4_xxyy_arr[i1, j2]) * (hy * hy) / 6.0

        f3_xxy_lo = (self.f2_xx_arr[i1, j2] - self.f2_xx_arr[i1, j1]) / hy + ((1.0 - 3.0 * ay * ay) * self.f4_xxyy_arr[
                    i1, j1] + (3.0 * by * by - 1) * self.f4_xxyy_arr[i1, j2]) * hy / 6.0
        f4_xxyy_lo = ay * self.f4_xxyy_arr[i1, j1] + by * self.f4_xxyy_arr[i1, j2]

        f_hi = ay * self.f_arr[i2, j1] + by * self.f_arr[i2, j2] + ((ay * ay * ay - ay) * self.f2_yy_arr[i2, j1] + (
               by * by * by - by) * self.f2_yy_arr[i2, j2]) * (hy * hy) / 6.0
        f_y_hi = (self.f_arr[i2, j2] - self.f_arr[i2, j1]) / hy + ((1.0 - 3.0 * ay * ay) * self.f2_yy_arr[i2, j1] + (
                 3.0 * by * by - 1) * self.f2_yy_arr[i2, j2]) * hy / 6.0

        f2_yy_hi = ay * self.f2_yy_arr[i2, j1] + by * self.f2_yy_arr[i2, j2]
        f2_xx_hi = ay * self.f2_xx_arr[i2, j1] + by * self.f2_xx_arr[i2, j2] + ((ay * ay * ay - ay) * self.f4_xxyy_arr[
                   i2, j1] + (by * by * by - by) * self.f4_xxyy_arr[i2, j2]) * (hy * hy) / 6.0

        f3_xxy_hi = (self.f2_xx_arr[i2, j2] - self.f2_xx_arr[i2, j1]) / hy + ((1.0 - 3.0 * ay * ay) * self.f4_xxyy_arr[
                    i2, j1] + (3.0 * by * by - 1) * self.f4_xxyy_arr[i2, j2]) * hy / 6.0
        f4_xxyy_hi = ay * self.f4_xxyy_arr[i2, j1] + by * self.f4_xxyy_arr[i2, j2]

        if f:
            f[0] = ax * f_lo + bx * f_hi + ((ax * ax * ax - ax) * f2_xx_lo + (bx * bx * bx - bx) * f2_xx_hi) * (
                    hx * hx) / 6.0
        if f_x:
            f_x[0] = (f_hi - f_lo) / hx + (
                    (1.0 - 3.0 * ax * ax) * f2_xx_lo + (3.0 * bx * bx - 1) * f2_xx_hi) * hx / 6.0
        if f_y:
            f_y[0] = ax * f_y_lo + bx * f_y_hi + ((ax * ax * ax - ax) * f3_xxy_lo + (bx * bx * bx - bx) * f3_xxy_hi) * (
                    hx * hx) / 6.0
        if f2_yy:
            f2_yy[0] = ax * f2_yy_lo + bx * f2_yy_hi + (
                    (ax * ax * ax - ax) * f4_xxyy_lo + (bx * bx * bx - bx) * f4_xxyy_hi) * (hx * hx) / 6.0


@njit
def CreateLQInterpolationKernel(x, N, x_arr, i1, i2, K, K1, K2):
    """A module intended to preprocess the variables of a call to the LQInterpolation method before full
       interpolation."""

    if x <= x_arr[0]:
        i0 = 0
    elif x >= x_arr[N - 1]:
        i0 = N - 1
    else:
        k1 = 0
        k2 = N - 1

        while (k2 - k1) > 1:
            l = (k1 + k2) >> 1
            if x_arr[l] > x:
                k2 = l
            else:
                k1 = l

        i0 = k1 if ((x - x_arr[k1]) < (x_arr[k2] - x)) else k2

    dx = x - x_arr[i0]

    if i0 == 0:
        i1[0] = 0
        i2[0] = 2
        dxl = x_arr[1] - x_arr[0]
        dxr = x_arr[2] - x_arr[1]
        K[0] = 1.0 - dx / dxl
        K[1] = dx / dxl
        K[2] = 0.0

        if K1:
            K1[0] = (2.0 * dx - 2.0 * dxl - dxr) / dxl / (dxl + dxr)
            K1[1] = -(2.0 * dx - dxr - dxl) / (dxr * dxl)
            K1[2] = (2.0 * dx - dxl) / dxr / (dxr + dxl)
    elif i0 == (N - 1):
        i1[0] = N - 3
        i2[0] = N - 1
        dxl = x_arr[N - 2] - x_arr[N - 3]
        dxr = x_arr[N - 1] - x_arr[N - 2]
        K[0] = 0.0
        K[1] = -dx / dxr
        K[2] = 1.0 + dx / dxr
        if K1:
            K1[0] = (2.0 * dx + dxr) / dxl / (dxr + dxl)
            K1[1] = -(2.0 * dx + dxr + dxl) / (dxr * dxl)
            K1[2] = (2.0 * dx + 2.0 * dxr + dxl) / dxr / (dxr + dxl)
    else:
        i1[0] = i0 - 1
        i2[0] = i0 + 1
        dxr = x_arr[i0 + 1] - x_arr[i0]
        dxl = x_arr[i0] - x_arr[i0 - 1]

        if dx > 0:
            K[0] = 0.0
            K[1] = 1.0 - dx / dxr
            K[2] = dx / dxr
        else:
            K[0] = -dx / dxl
            K[1] = 1.0 + dx / dxl
            K[2] = 0.0
        if K1:
            K1[0] = (2.0 * dx - dxr) / dxl / (dxr + dxl)
            K1[1] = -(2.0 * dx - dxr + dxl) / (dxr * dxl)
            K1[2] = (2.0 * dx + dxl) / dxr / (dxr + dxl)

    if K2:
        K2[0] = 2.0 / dxl / (dxr + dxl)
        K2[1] = -2.0 / (dxr * dxl)
        K2[2] = 2.0 / dxr / (dxr + dxl)


@njit
def LQInterpolate(x, N, x_arr, y_arr, y, y1):
    """An interpolation method whose calculation is based on linear quartile handling."""

    K = np.full(3, bigNeg)
    K1 = np.full(3, bigNeg)
    i1 = 0
    i2 = 0

    CreateLQInterpolationKernel(x, N, x_arr, i1, i2, K, K1, 0)

    y[0] = 0.0
    y1[0] = 0.0
    for i in range(i1, i2 + 1):
        y[0] += (y_arr[i] * K[i - i1])
        y1[0] += (y_arr[i] * K1[i - i1])


@njit
def LQInterpolate2D(x, y, Nx, Ny, x_arr, y_arr, f_arr, f, f_x, f_y, f2_yy):
    """A modified version of the LQInterpolate method which is capable of interpolation with two-dimensional arrays."""

    Kx = np.full(3, bigNeg)
    Ky = np.full(3, bigNeg)
    Kx1 = np.full(3, bigNeg)
    Ky1 = np.full(3, bigNeg)
    Ky2 = np.full(3, bigNeg)

    i1 = 0
    i2 = 0
    j1 = 0
    j2 = 0

    CreateLQInterpolationKernel(x, Nx, x_arr, i1, i2, Kx, Kx1 if f_x else 0, 0)
    CreateLQInterpolationKernel(y, Ny, y_arr, j1, j2, Ky, Ky1 if f_y else 0, Ky2 if f2_yy else 0)

    if f:
        f[0] = 0.0
    if f_x:
        f_x[0] = 0.0
    if f_y:
        f_y[0] = 0.0
    if f2_yy:
        f2_yy[0] = 0.0

    for i in range(i1, i2 + 1):
        for j in range(j1, j2 + 1):
            if f:
                f[0] += (f_arr[i, j] * Kx[i - i1] * Ky[j - j1])
            if f_x:
                f_x[0] += (f_arr[i, j] * Kx1[i - i1] * Ky[j - j1])
            if f_y:
                f_y[0] += (f_arr[i, j] * Kx[i - i1] * Ky1[j - j1])
            if f2_yy:
                f2_yy[0] += (f_arr[i, j] * Kx[i - i1] * Ky2[j - j1])


@njit
def IntTabulated(x, y, N):
    """Integrates a tabulated set of data on a closed interval, using a five-point Newton-Cotes integration formula."""

    s = 0.0

    for i in range(0, N - 1):
        s += (y[i] + y[i + 1]) * math.fabs(x[i + 1] - x[i]) / 2

    return s


@njit
def IntTabulatedLog(t, u, N):
    """A modified version of the IntTabulated method that integrates over the tabulated set using a logarithmic
       formula."""

    s = 0.0

    for i in range(0, N - 1):
        s += (math.exp(t[i + 1] + u[i + 1]) - math.exp(t[i] + u[i])) / ((t[i + 1] + u[i + 1]) - (t[i] + u[i])) * (
                t[i + 1] - t[i])
    return s


@njit
def bessi0(x):
    """Returns the 0th order I bessel function for argument x."""

    ax = math.fabs(x)
    if ax < 3.75:
        y = x / 3.75
        y *= y
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y
              * 0.45813e-2)))))
    else:
        y = 3.75 / ax
        ans = (math.exp(ax) / math.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 +
              y * (0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))))

    return ans


@njit
def bessi1(x):
    """Returns the 1st order I bessel function for argument x."""

    ax = math.fabs(x)
    if ax < 3.75:
        y = x / 3.75
        y *= y
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2
              + y * 0.32411e-3))))))
    else:
        y = 3.75 / ax
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
        ans *= (math.exp(ax) / math.sqrt(ax))

    return -ans if (x < 0.0) else ans


@njit
def expbessk0(x):
    """Returns the exponential 0th order K bessel function for argument x."""

    if x <= 2.0:
        y = x * x / 4.0
        ans = ((-math.log(x / 2.0) * bessi0(x)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.3488590e-1 +
               y * (0.262698e-2 + y * (0.10750e-3 + y * 0.74e-5))))))) * math.exp(x)
    else:
        y = 2.0 / x
        ans = (1.0 / math.sqrt(x)) * (1.25331414 + y * (-0.7832358e-1 + y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (
              0.587872e-2 + y * (-0.251540e-2 + y * 0.53208e-3))))))

    return ans


@njit
def expbessk1(x):
    """Returns the exponential 1st order K bessel function for argument x."""

    if x <= 2.0:
        y = x * x / 4.0
        ans = ((math.log(x / 2.0) * bessi1(x)) + (1.0 / x) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y
              * (-0.18156897 + y * (-0.1919402e-1 + y * (-0.110404e-2 + y * (-0.4686e-4)))))))) * math.exp(x)
    else:
        y = 2.0 / x
        ans = (1.0 / math.sqrt(x)) * (1.25331414 + y * (0.23498619+ y * (-0.3655620e-1 + y * (0.1504268e-1 + y
              * (-0.780353e-2 + y * (0.325614e-2 + y * (-0.68245e-3)))))))

    return ans


@njit
def ExpBesselK(n, x):
    """Returns the exponential of x multiplied by the nth kernel of x (K bessel function)

       (exp(x) * K_n(x))"""

    tox = 2.0 / x
    bkm = expbessk0(x)
    bk = expbessk1(x)

    for j in range(1, n):
        bkp = bkm + j * tox * bk
        bkm = bk
        bk = bkp

    return bk


# Adapted from: http://astro.uni-tuebingen.de/software/idl/astrolib/math/polint.html
@njit
def polint(xa, ya, n, x, y, dy):
    """
    Interpolate a set of n points by fitting a polynomial of degree N-1

    :param xa: X Numeric vector, all values must be distinct.   The number of values in XA should rarely exceed 10
    (i.e. a 9th order polynomial)
    :param ya: Y Numeric vector, same number of elements
    :param n: control the degree of the polynomial
    :param x: Numeric scalar specifying value to be interpolated
    :param y: Scalar, interpolated value in (XA,YA) corresponding to X
    :param dy: Error estimate on Y, scalar

    """

    ns = 1

    c = np.full(10, bigNeg)
    d = np.full(10, bigNeg)

    dif = math.fabs(x - xa[1])

    for i in range(1, n + 1):

        dift = math.fabs(x - xa[i])

        if dift < dif:
            ns = i
            dif = dift

        c[i] = ya[i]
        d[i] = ya[i]

    y[0] = ya[ns]
    ns -= 1

    for m in range(1, n):
        for i in range(1, (n - m) + 1):
            ho = xa[i] - x
            hp = xa[i + m] - x
            w = c[i + 1] - d[i]
            den = ho - hp
            den = w / den
            d[i] = hp * den
            c[i] = ho * den

        dy[0] = (c[ns+1] if 2*ns < (n-m) else d[ns])
        ns -= 1
        y[0] += dy[0]

# @njit
# def bitwise_it(n_, it_):
#     for j in range(1, (n_ - 1)):
#         it_ <<= 1
#
#     return it_


# Adapted from http://astro.uni-tuebingen.de/software/idl/astrolib/math/trapzd.html
# @njit #Note not needed
def trapzdQ(F, a, b, n, s):
    """
    Compute the nth stage of refinement of an extended trapezoidal rule.

    :param F: The function to be integrated
    :param a: First limit of integration
    :param b: Second limit of integration
    :param n: The number of points at which to compute the function for the current iteration
    :param s: The total sum from the previous iterations on input and the refined sum after the current iteration on
              output
    """
    if n == 1:
        s[0] = 0.5 * (b - a) * (F[0].F(a) + F[0].F(b))
        return s[0]

    else:
        it = 1

        # it = bitwise_it(n, it)

        for j in range(1, (n - 1)):
            it <<= 1

        tnm = it
        de = (b - a) / tnm
        x = a + 0.5 * de

        sum = 0.0

        for j in range(1, it + 1):
            sum += F[0].F(x)
            x += de

        s[0] = 0.5 * (s[0] + (b - a) * sum / tnm)
        return s[0]


JMAXP = JMAX + 1
romK = 6


# Adapted from https://www.l3harrisgeospatial.com/docs/
# qromb.html#:~:text=The%20QROMB%20function%20evaluates%20the,%2C%20B%5D%20using%20Romberg%20integration.
# @njit
def qromb(F, a, b, EPS, err):
    """
    Evaluates the integral of a function over the closed interval [a, b] using Romberg integration.

    :param F: The function to be integrated
    :param a: First limit of integration
    :param b: Second limit of integration
    :param EPS: The desired fractional accuracy.
    :param err: error flag
    """

    err[0] = 0

    ts = np.array([bigNeg])
    ss = np.array([bigNeg])
    dss = np.array([bigNeg])

    s = np.full(JMAXP, bigNeg)
    h = np.full((JMAXP + 1), bigNeg)

    h[1] = 1.0

    for j in range(1, (JMAX + 1)):

        s[j] = trapzdQ(F, a, b, j, ts)

        if j >= romK:

            h = np.roll(h, -(j - romK))
            s = np.roll(s, -(j - romK))

            polint(h, s, romK, 0.0, ss, dss)

            h = np.roll(h, j - romK)
            s = np.roll(s, j - romK)

            if (math.fabs(dss[0])) <= EPS * (math.fabs(ss[0])) or not (np.isfinite(ss[0])):
                return ss[0]

        h[j + 1] = 0.25 * h[j]

    err[0] = 1
    return ss[0]


# Adapted from http://astro.uni-tuebingen.de/software/idl/astrolib/math/trapzd.html
# @njit
def trapzd(F, a, b, N):
    """
    Compute the nth stage of refinement of an extended trapezoidal rule.

    :param F: The function to be integrated
    :param a: first limit of integration
    :param b: second limit of integration
    :param N: The number of points at which to compute the function for the current iteration
    """
    s = 0.0
    x = a
    dx = (b - a) / N

    for i in range(0, (N + 1)):
        u = F[0].F(x)

        if (i == 0) or (i == N):
            u /= 2
        s += u
        x += dx

    return s * dx


# specIntFunLog = [
#     ('oldF', nb.pyobject)
# ]
#
#
# @jitclass(specIntFunLog)
class IntegrableFunctionLog:
    """Kernel for handling the setting a function object on a logarithmic scale."""

    def __init__(self):
        self.oldF = np.array([None])

    def F(self, t):
        x = math.exp(t)
        return self.oldF[0].F(x) * x


@njit
def qrombLog(F, a, b, EPS, err):
    """Calls the qromb romberg integration method on a function object f modified onto a logarithmic scale.

    :param F: The function to be integrated
    :param a: First limit of integration
    :param b: Second limit of integration
    :param EPS: The desired fractional accuracy.
    :param err: error flag
    """

    ifl = np.array([IntegrableFunctionLog()])
    ifl[0].oldF[0] = F[0]

    return qromb(ifl, math.log(a), math.log(b), EPS, err)


# @njit
def trapzdLog(F, a, b, N):
    """Calls the trapzd refinement method on a function object f modified onto a logarithmic scale.

    :param F: The function to be integrated
    :param a: first limit of integration
    :param b: second limit of integration
    :param N: The number of points at which to compute the function for the current iteration
    """

    ifl = np.array([IntegrableFunctionLog()])
    ifl[0].oldF[0] = F[0]

    return trapzd(ifl, math.log(a), math.log(b), N)


@njit
def InterpolateBilinear(arr, i1, i2, N1, N2, missing):
    """Interpolation on an equidistant grid.

       i1 and i2 are the fractional indices of the required point."""

    if i1 < 0 or i1 > (N1 - 1) or i2 < 0 or i2 > (N2 - 1):
        return missing

    j = int(i1)
    k = int(i2)
    t = i1 - j
    u = i2 - k

    y1 = arr[N2 * j + k]
    y2 = arr[N2 * (j + 1) + k]
    y3 = arr[N2 * (j + 1) + k + 1]
    y4 = arr[N2 * j + k + 1]

    return (1.0 - t) * (1.0 - u) * y1 + t * (1.0 - u) * y2 + t * u * y3 + (1.0 - t) * u * y4


@njit
def InterpolBilinear(arr, x1arr, x2arr, x1, x2, N1, N2):
    """Interpolation on an arbitrary grid.

       Performs extrapolation, if the point is outside the range. """

    if x1 < x1arr[0]:
        j = 0
        j1 = 1
    elif x1 > x1arr[N1 - 1]:
        j = N1 - 2
        j1 = N1 - 1
    else:
        j = 0
        j1 = N1 - 1
        while (j1 - j) > 1:
            l = (j1 + j) >> 1
            if x1arr[l] > x1:
                j1 = l
            else:
                j = l

    dx1 = x1arr[j1] - x1arr[j]
    t = (x1 - x1arr[j]) / dx1

    if x2 < x2arr[0]:
        k = 0
        k1 = 1

    elif x2 > x2arr[N2 - 1]:
        k = N2 - 2
        k1 = N2 - 1
    else:
        k = 0
        k1 = N2 - 1
        while (k1 - k) > 1:
            l = (k1 + k) >> 1
            if x2arr[l] > x2:
                k1 = l
            else:
                k = l

    dx2 = x2arr[k1] - x2arr[k]
    u = (x2 - x2arr[k]) / dx2

    y1 = arr[N2 * j + k]
    y2 = arr[N2 * j1 + k]
    y3 = arr[N2 * j1 + k1]
    y4 = arr[N2 * j + k1]

    return (1.0 - t) * (1.0 - u) * y1 + t * (1.0 - u) * y2 + t * u * y3 + (1.0 - t) * u * y4


@njit
def ErfC(x):
    """Returns the complementary error function"""

    z = math.fabs(x)
    t = 1.0 / (1.0 + 0.5 * z)

    ans = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t
          * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))

    return ans if (x >= 0.0) else (2.0 - ans)


# @njit
def SecantRoot(F, x1, x2, EPS):
    """
    A method which uses a succession of roots of secant lines to better approximate a root of function f

    :param F: The function to be
    :param x1: The first x coordinate
    :param x2: The second x coordinate
    :param EPS: The desired fractional accuracy
    """

    MAXIT = 20

    fl = F[0].F(x1)
    f = F[0].F(x2)

    if (math.fabs(fl)) < (math.fabs(f)):
        rts = x1
        xl = x2
        swap = fl
        fl = f
        f = swap
    else:
        xl = x1
        rts = x2

    j = 0

    while True:
        dx = (xl - rts) * f / (f - fl)
        xl = rts
        fl = f
        rts += dx
        f = F[0].F(rts)
        j += 1

        if not((math.fabs(dx)) > EPS and f != 0.0 and j < MAXIT):
            break

    return rts if (j < MAXIT) else dNaN


# @njit
def BrentRoot(F, x1, x2, tol):
    """
    A hybrid root-finding algorithm combining the bisection, secant, and inverse quadratic interpolation methods.

    :param F: The function to be
    :param x1: The first x coordinate
    :param x2: The second x coordinate
    :param tol: The desired fractional accuracy
    """

    a = x1
    b = x2
    c = x2
    d = 0.0
    e = 0.0

    fa = F[0].F(a)
    fb = F[0].F(b)

    BrentMAXIT = 100

    if not (np.isfinite(fa)) or not (np.isfinite(fb)):
        return dNaN

    if fa * fb > 0.0:
        return dNaN
    else:
        fc = fb
        for iter in range(1, BrentMAXIT + 1):
            if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
                c = a
                fc = fa
                e = b - a
                d = b - a

            if (fc if (fc > 0) else -fc) < (fb if (fb > 0) else -fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa

            tol1 = 0.5 * tol
            xm = 0.5 * (c - b)

            if (math.fabs(xm)) <= tol1 or fb == 0.0:
                return b
            if (math.fabs(e)) >= tol1 and (math.fabs(fa)) > (math.fabs(fb)):

                s = fb / fa
                if a == c:
                    p = 2.0 * xm * s
                    q = 1.0 - s
                else:
                    q = fa / fc
                    r = fb / fc
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)

                if p > 0.0:
                    q = -q

                p = math.fabs(p)

                min1 = 3.0 * xm * q - (math.fabs(tol1 * q))
                min2 = math.fabs(e * q)

                if 2.0 * p < (min1 if min1 < min2 else min2):
                    e = d
                    d = p / q
                else:
                    d = xm
                    e = d
            else:
                d = xm
                e = d
            a = b
            fa = fb

            if math.fabs(d) > tol1:
                b += d
            else:
                b += math.fabs(tol1) if xm >= 0.0 else -math.fabs(tol1)

            fb = F[0].F(b)
            if not (np.isfinite(fb)):
                return dNaN

        return dNaN


@njit
def Erf(x):
    """Returns the error function"""

    return 1.0 - ErfC(x)


@njit
def bessj0(x):
    """Returns the 0th order J bessel function for argument x."""

    ax = math.fabs(x)
    if ax < 8.0:
        y = x * x
        ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y
               * (-184.9052456)))))
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
        ans = ans1 / ans2
    else:
        z = 8.0 / ax
        y = z * z
        xx = ax - 0.785398164
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y
               * 0.934935152e-7)))
        ans = math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)

    return ans


@njit
def bessj1(x):
    """Returns the 1st order J bessel function for argument x."""

    ax = math.fabs(x)
    if ax < 8.0:
        y = x * x
        ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y
               * (-30.16036606))))))
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        ans = ans1 / ans2
    else:
        z = 8.0 / ax
        y = z * z
        xx = ax - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y
               * 0.105787412e-6)))
        ans = math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)
        if x < 0.0:
            ans = -ans

    return ans


@njit
def FindBesselJ(x, n, Js, Js1):
    """The exact expression for the bessel function calculations."""

    ACC = 40.0
    BIGNO = 1.0e10
    BIGNI = 1.0e-10

    if n == 1:
        Bsn = bessj1(x)
        Bsn1 = bessj0(x)
    else:
        ax = math.fabs(x)
        if ax > float(n):
            tox = 2.0 / ax
            bjm = bessj0(ax)
            bj = bessj1(ax)

            for j in range(1, n):
                bjp = tox * bj * j - bjm
                bjm = bj
                bj = bjp

            Bsn = bj
            Bsn1 = bjm

        else:
            tox = 2.0 / ax
            m = 2 * int((n + int(math.sqrt(ACC * n))) / 2)
            jsum = 0

            bjp = 0.0
            Bsn = 0.0
            Bsn1 = 0.0
            sum = 0.0

            bj = 1.0
            for j in range(m, 0, -1):
                bjm = tox * bj * j - bjp
                bjp = bj
                bj = bjm

                watchIter = 0

                while (math.fabs(bj)) > BIGNO:

                    watchIter += 1

                    bj *= BIGNI
                    bjp *= BIGNI
                    Bsn *= BIGNI
                    Bsn1 *= BIGNI
                    sum *= BIGNI

                if jsum:
                    sum += bj

                jsum = jsum == 0

                if j == n:
                    Bsn = bjp
                if j == (n - 1):
                    Bsn1 = bjp

            sum = 2.0 * sum - bj
            Bsn /= sum
            Bsn1 /= sum

            if n == 2:
                Bsn1 = bessj1(x)

    Js[0] = Bsn
    Js1[0] = Bsn1 - Bsn * n / x

    if x < 0.0:
        if n & 1:
            Js[0] = -(Js[0])
        else:
            Js1[0] = -(Js[0])

    return


@njit
def FindBesselJ_WH(Sx, S, JS, JS1):
    """The approximate Wild and Hill expression for the bessel function calculations."""

    A = 0.503297
    B = 1.193000

    p16 = 1.0 / 6
    pm23 = -2.0 / 3

    x = Sx / S
    t1 = math.sqrt(1.0 - (x ** 2))
    t2 = t1 * t1 * t1
    a = math.pow(t2 + A / S, p16)
    b = math.pow(t2 + B / S, p16) * (1.0 - 0.2 * math.pow(S, pm23))
    F = a * b
    Z = x * math.exp(t1) / (1.0 + t1)

    JS[0] = math.pow(Z, S) / math.sqrt(2.0 * math.pi * S) / a
    JS1[0] = F * (JS[0]) / x


@njit
def LnGamma(xx):
    """Returns the logarithm of the gamma function of the xx argument."""

    cof = np.array([76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2,
                    -0.5395239384953e-5])

    y = xx
    x = xx

    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015

    for j in range(0, 6):
        ser += cof[j] / ++y
    return -tmp + math.log(2.5066282746310005 * ser / x)


@njit
def Gamma(z):
    """Returns the gamma function of the z argument by finding the exponent the LnGamma function."""

    return math.exp(LnGamma(z))
