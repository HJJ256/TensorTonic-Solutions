import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    from collections import Counter
    label_counter = Counter(y_train)
    preds = []
    for row in X_test:
        preds.append(label_counter.most_common(1)[0][0])
    return preds