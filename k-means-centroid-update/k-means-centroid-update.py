def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    import numpy as np
    points = np.array(points)
    assignments = np.array(assignments)
    unique_counts = np.zeros((k,))
    centroids = np.zeros((k,points.shape[1]))
    for i in range(len(points)):
        cluster_idx = assignments[i]
        unique_counts[cluster_idx]+=1
        centroids[cluster_idx] += points[i]
    print(centroids)
    centroids =  [(centroids[i]/unique_counts[i]).tolist() if unique_counts[i]>0 else centroids[i].tolist() for i in range(k)]
    return centroids