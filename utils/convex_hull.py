import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import math


def get_convex_hull_area(points):
    hull = ConvexHull(points)
    return hull.volume

A = []
if __name__ == '__main__':
    rng = np.random.default_rng()
    points = rng.random((30, 2))  # 30 random points in 2-D

    hull = ConvexHull(points)

    A.append(hull.volume)
    # print(A[-1])
    # print(np.mean(A))
    # print(np.std(A))


    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


    plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    plt.show()
