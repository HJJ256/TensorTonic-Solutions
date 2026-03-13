import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.array(X) #NxD
    X_bar = np.mean(X, axis=0) # (D,) mean for each feature
    X_c = X-X_bar #N,D X_bar gets auto broadcasted
    N,_= X_c.shape
    C = (X_c.T.dot(X_c))/(N-1)
    eig_val, eig_vec = np.linalg.eig(C)
    idxs = eig_val.argsort()[::-1]
    eig_val = eig_val[idxs]
    eig_vec = eig_vec[:,idxs]
    W = eig_vec[:,:k]
    return X_c@W
    