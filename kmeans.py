import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
import os
import shutil
from PIL import Image
import glob

centercolor = '#e60e0e'


def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def label_points(points, centers, k):
    labels = np.zeros((len(points)))
    for p_index in range(len(points)):
        distances = [dist(points[p_index], centers[i]) for i in range(k)]      
        labels[p_index] = (distances.index(min(distances)))
    
    return labels


def recompute_centers(points, labels, old_centers, k):
    cluster_dict = {i:[] for i in range(k)}
    for p, l in zip(points, labels):
        cluster_dict[l].append(p)

    centers = np.zeros((k, 2))
    for i in range(k):
        if len(cluster_dict[i]) > 0:
            centers[i][0] = np.mean([p[0] for p in cluster_dict[i]])
            centers[i][1] = np.mean([p[1] for p in cluster_dict[i]])
        else:
            centers[i][0] = old_centers[i][0]
            centers[i][1] = old_centers[i][1]

    return centers

def plot_points(points, centers, labels, xmin, xmax, ymin, ymax, filename, title, plot_centers=True):
    xpoints = [p[0] for p in points] + ([xmin - abs(xmin / 10)] * len(centers))
    ypoints = [p[1] for p in points] + ([0] * len(centers))
    xcenters = [p[0] for p in centers]
    ycenters = [p[1] for p in centers]
    labels = np.concatenate((labels, np.array(range(len(centers)))))

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    if plot_centers:
        plt.scatter(xpoints, ypoints, s=40, c=labels)
        plt.scatter(xcenters, ycenters, s=80, c=centercolor, marker='^')
    else:
        plt.scatter(xpoints, ypoints, s=40)
    
    plt.title(title, loc='left')
    plt.savefig(filename)
    plt.clf()


def has_converged(old_centers, new_centers):
    for old, new in zip(old_centers, new_centers):
        if not (old == new).all():
            return False
    return True

def animate(points, centers):
    old_centers = centers + 1

    pad = 1.1
    xmin = np.min(points[:,0]) * pad
    xmax = np.max(points[:,0]) * pad
    ymin = np.min(points[:,1]) * pad
    ymax = np.max(points[:,1]) * pad

    k = len(centers)

    if os.path.exists('frames'):
        shutil.rmtree('frames')
    os.makedirs('frames')
 
    plot_points(points, centers, [], xmin, xmax, ymin, ymax, 'frames/0', 'Data Points', False)
    
    index = 1
    epoch = 1
    labels = label_points(points, centers, k)
    while not has_converged(old_centers, centers):
        old_centers = centers
        centers = recompute_centers(points, labels, old_centers, k)
        plot_points(points, centers, labels, xmin, xmax, ymin, ymax, 'frames/' + str(index), 'Epoch: ' + str(epoch))
        index += 1
        labels = label_points(points, centers, k)
        plot_points(points, centers, labels, xmin, xmax, ymin, ymax, 'frames/' + str(index), 'Epoch: ' + str(epoch))
        index += 1
        plot_points(points, centers, labels, xmin, xmax, ymin, ymax, 'frames/' + str(index), 'Epoch: ' + str(epoch))
        index += 1
        epoch += 1
        

    for i in range(6):
        plot_points(points, centers, labels, xmin, xmax, ymin, ymax, 'frames/' + str(index + i), 'Epoch: ' + str(epoch - 1) + ' '*8 + 'Done!')

    frames = []
    images = sorted(glob.glob('frames/*.png'), key=os.path.getmtime)
    for img in images:
        frames.append(Image.open(img))

    frames[0].save('kmeans_animation.gif', format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)
    shutil.rmtree('frames')


# points = np.array([(4, 4), (-3, 2), (-1, -1), (2, 6), (-3, -3)])
# centers = np.array([[0, -1], [0, 1]])

points, labels = datasets.make_blobs(n_samples=500, n_features=2, centers=5, random_state=2)
centers = np.array([[0, -1], [0, 1], [4, -1], [2, 3], [5, -4], [3, 3]])

animate(points, centers)
