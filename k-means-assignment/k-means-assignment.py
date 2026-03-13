import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.array(points)
    centroids = np.array(centroids)
    assignments = []
    for point in points:
        distances = np.linalg.norm(point-centroids, axis=1)
        cluster_idx = distances.argmin()
        assignments.append(int(cluster_idx))
    return assignments