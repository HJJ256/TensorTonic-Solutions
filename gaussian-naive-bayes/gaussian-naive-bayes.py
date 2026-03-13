import numpy as np
def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    unique_labels, unique_counts = np.unique(y_train, return_counts=True) 
    p_c = unique_counts/len(y_train)
    mus = []
    sigmas = []
    for label, count_label in zip(unique_labels, unique_counts):
        feats = X_train[y_train==label]
        mu_c = np.sum(feats,axis=0)/count_label
        sigma_2 = (np.sum((feats - mu_c)**2,axis=0)/count_label) + 1e-9
        mus.append(mu_c)
        sigmas.append(sigma_2)

    y_pred = []
    for feats in X_test:
        probs = []
        for idx, label, mu_c, sigma_2 in zip(range(len(unique_labels)), unique_labels, mus, sigmas):
            log_p_x_c = -0.5*np.log(2*np.pi*sigma_2)-(((feats-mu_c)**2)/(2*sigma_2))
            p_c_giv_x = np.log(p_c[idx]) + np.sum(log_p_x_c)
            probs.append(p_c_giv_x)
        y_pred.append(np.argmax(probs))
    return y_pred
        