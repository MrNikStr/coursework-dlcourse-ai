import numpy as np


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    assert prediction.shape == ground_truth.shape
    assert prediction.ndim == 1

    correct = np.sum(prediction == ground_truth)
    total = prediction.shape[0]

    return correct / float(total)
