import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    neighbors = []
    if len(X_test)==0:
        return np.array(neighbors, dtype=int).reshape((0,k))
    X_train = np.array(X_train)
    X_train = X_train.reshape((len(X_train),-1))
    X_test = np.array(X_test)
    X_test = X_test.reshape((len(X_test),-1))
    for feats in X_test:
        distances = np.linalg.norm(X_train-feats, axis=1)
        idxs = np.full((k,), -1)
        idxs[:min(k,len(distances))] = distances.argsort()[:k]
        neighbors.append(idxs)
    return np.array(neighbors)