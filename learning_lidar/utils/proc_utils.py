import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def make_interpolated_image(n_samples, im):
    """
    Randomly sampe and interpolate an image.
    :param n_samples: int object. Number of samples to make.
    :param im: np.array. Input 2D image
    :return: interp_im : np.array. A 2D image interpolated from a random selection of pixels, of the original image im.
    """
    nx, ny = im.shape[1], im.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    ix = np.random.randint(im.shape[1], size=n_samples)
    iy = np.random.randint(im.shape[0], size=n_samples)
    samples = im[iy, ix]
    interp_im = griddata((iy, ix), samples, (Y, X), method='nearest', fill_value=0)
    return interp_im


def normalize(x, max_value=1):
    """
    :param x: np.array. Input signal to normalize. can be 1D, 2D ,3D ...
    :param max_value: np.float. The max number to normalize the signal
    :return: Normalized signal
    """
    return max_value * (x - x.min()) / (x.max() - x.min())


# Bezier
# This module is based on :
# https://towardsdatascience.com/b%C3%A9zier-interpolation-8033e9a262c2
# find the a & b points

def get_bezier_coef(points):
    """
    Calculates Bezier coefficients for n points, having n-1 sections (one section between 2 consecutive points).
    The matrix rank is of size of number of sections, a.k.a n-1.
    :param points:
    :return:
    """

    n = len(points) - 1

    # build coefficients matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B


def get_cubic(a, b, c, d):
    """
    Returns the general Bezier cubic formula given 4 control points
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c \
                     + np.power(t, 3) * d


def get_bezier_cubic(points):
    """
    Returns one cubic curve for each consecutive points
    :param points:
    :return:
    """
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]


def evaluate_bezier(points, n):
    """
    Evaluate each cubic curve on the range [0, 1] sliced in n points.
    Note that the last section should include the last point, therefore last section is added.
    :param points:
    :param n:
    :return:
    """
    curves = get_bezier_cubic(points)
    path = np.array([fun(t) for fun in curves[0:-1] for t in np.linspace(0, 1, n)])
    last_section = np.array([curves[-1](t) for t in np.linspace(0, 1, n + 1)])
    total_path = np.append(path, last_section, axis=0)
    return total_path


def bezier_example_usage():
    # generate 5 (or any number that you want) random points that we want to fit (or set them yourself)
    points = np.random.rand(9, 2)
    points[:, 0] *= 10
    points[:, 1] *= 300
    # fit the points with Bezier interpolation
    # use 50 points between each consecutive points to draw the curve
    path = evaluate_bezier(points, 50)

    # extract x & y coordinates of points
    x, y = points[:, 0], points[:, 1]
    px, py = path[:, 0], path[:, 1]

    # plot
    plt.figure(figsize=(11, 8))
    plt.plot(px, py, 'b-')
    plt.plot(x, y, 'ro')
    plt.show()
