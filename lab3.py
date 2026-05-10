import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("uber-raw-data-sep14.csv")
df = df.sample(2000)

data = df[['Lat', 'Lon']].values

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data_norm = (data - mean) / std

UNVISITED = 0
NOISE = -1

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def region_query(data, point_idx, eps):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_idx], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, minPts, core_points):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        n_point = neighbors[i]
        if labels[n_point] == UNVISITED:
            labels[n_point] = cluster_id
            n_neighbors = region_query(data, n_point, eps)
            if len(n_neighbors) >= minPts:
                core_points.add(n_point)
                neighbors += n_neighbors
        if labels[n_point] == NOISE:
            labels[n_point] = cluster_id
        i += 1

def dbscan(data, eps, minPts):
    labels = [UNVISITED] * len(data)
    cluster_id = 0
    core_points = set()
    for i in range(len(data)):
        if labels[i] != UNVISITED:
            continue
        neighbors = region_query(data, i, eps)
        if len(neighbors) < minPts:
            labels[i] = NOISE
        else:
            cluster_id += 1
            core_points.add(i)
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, minPts, core_points)
    return np.array(labels), core_points

def visualize_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            color = 'black'
            marker = 'x'
            name = 'Noise'
        else:
            color = None
            marker = 'o'
            name = f'Cluster {label}'
        points = data[labels == label]
        plt.scatter(points[:, 0], points[:, 1], c=color, label=name, marker=marker)
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_point_types(data, labels, core_points, eps, minPts):
    core = []
    border = []
    noise = []
    for i in range(len(data)):
        if labels[i] == -1:
            noise.append(data[i])
        elif i in core_points:
            core.append(data[i])
        else:
            border.append(data[i])
    core = np.array(core)
    border = np.array(border)
    noise = np.array(noise)
    plt.figure(figsize=(8, 6))
    if len(core) > 0:
        plt.scatter(core[:, 0], core[:, 1], c='green', label='Core')
    if len(border) > 0:
        plt.scatter(border[:, 0], border[:, 1], c='blue', label='Border')
    if len(noise) > 0:
        plt.scatter(noise[:, 0], noise[:, 1], c='red', label='Noise')
    plt.title(f'Point Types (eps={eps}, minPts={minPts})')
    plt.legend()
    plt.show()

eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
minPts_values = [3, 5, 10]

for eps in eps_values:
    for minPts in minPts_values:
        print(f"Running DBSCAN with eps={eps}, minPts={minPts}")
        labels, core_points = dbscan(data_norm, eps, minPts)
        visualize_clusters(data_norm, labels, f'Clusters (eps={eps}, minPts={minPts})')
        visualize_point_types(data_norm, labels, core_points, eps, minPts)