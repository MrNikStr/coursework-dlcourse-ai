import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
        prediction, np array of int (num_samples) - model predictions
        ground_truth, np array of int (num_samples) - true labels

    Returns:
        precision, recall, f1, accuracy - classification metrics
    """

    tp = float(np.sum((prediction == 1) & (ground_truth == 1)))
    tn = float(np.sum((prediction == 0) & (ground_truth == 0)))
    fp = float(np.sum((prediction == 1) & (ground_truth == 0)))
    fn = float(np.sum((prediction == 0) & (ground_truth == 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


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
