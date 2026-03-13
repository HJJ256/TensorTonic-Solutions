import numpy as np
from collections import Counter
def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.array(predictions).T
    preds = []
    for sample_pred in predictions:
        vote_counter = Counter(sample_pred)
        preds.append(sorted(vote_counter.items(), key=lambda x: (-x[1],x[0]))[0][0])
    return preds