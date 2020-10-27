import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets

centercolor = '#000000'
colordict = {
    0: '#b30000',
    1: '#005cb3',
    2: '#fcc600'
}


def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def label_points(points, centers, k):
    labels = np.zeros((len(points)))
    for p_index in range(len(points)):
        distances = [dist(points[p_index], centers[i]) for i in range(k)]      
        labels[p_index] = (distances.index(min(distances)))
    
    return labels


def recompute_centers(points, labels, k):
    cluster_dict = {i:[] for i in range(k)}
    for p, l in zip(points, labels):
        cluster_dict[l].append(p)

    centers = np.zeros((k, 2))
    for i in range(k):
        centers[i][0] = np.mean([p[0] for p in cluster_dict[i]])
        centers[i][1] = np.mean([p[1] for p in cluster_dict[i]])

    return centers

def plot_points(points, centers, labels, k):
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xcenters = [p[0] for p in centers]
    ycenters = [p[1] for p in centers]

    plt.scatter(xpoints, ypoints, s=20, c=labels)
    plt.scatter(xcenters, ycenters, s=60, c=centercolor, marker='^')
    plt.show()

def has_converged(old_centers, new_centers):
    for old, new in zip(old_centers, new_centers):
        print('testing:', old, '==', new)
        if not (old == new).all():
            return False
    return True

points = np.array([(4, 4), (-3, 2), (-1, -1), (2, 6), (-3, -3)])
centers = np.array([[0, -1], [0, 1], [4, -1], [2, 3], [5, -4], [3, 3]])
old_centers = centers + 1

points, labels = datasets.make_blobs(n_samples=500, n_features=2, centers=5, random_state=2)

k = len(centers)

while not has_converged(old_centers, centers):
    labels = label_points(points, centers, k)
    plot_points(points, centers, labels, k)
    old_centers = centers
    centers = recompute_centers(points, labels, k)
