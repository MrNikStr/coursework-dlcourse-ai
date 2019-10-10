import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    raise Exception('Not implemented')
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    assert prediction.shape == ground_truth.shape
    assert prediction.ndim == 1

    correct = np.sum(prediction == ground_truth)
    total = prediction.shape[0]

    return correct / float(total)
