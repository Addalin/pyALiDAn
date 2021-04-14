import numpy as np
import matplotlib.pyplot as plt
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
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


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
    path  = np.array([fun(t) for fun in curves[0:-1] for t in np.linspace(0, 1, n)])
    last_section = np.array([curves[-1](t) for t in np.linspace(0, 1, n+1)])
    total_path = np.append(path,last_section,axis=0)
    return total_path

def main():
    # generate 5 (or any number that you want) random points that we want to fit (or set them yourself)
    points = np.random.rand(9, 2)
    points[:,0]*=10
    points [ : , 1 ] *= 300
    # fit the points with Bezier interpolation
    # use 50 points between each consecutive points to draw the curve
    path = evaluate_bezier(points, 50)

    # extract x & y coordinates of points
    x, y = points[:,0], points[:,1]
    px, py = path[:,0], path[:,1]

    # plot
    plt.figure(figsize=(11, 8))
    plt.plot(px, py, 'b-')
    plt.plot(x, y, 'ro')
    plt.show()

if __name__ == '__main__' :
    main()