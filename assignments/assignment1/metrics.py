def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = sum(1 if i == 1 and j == 1 else 0 for i, j in zip(prediction, ground_truth))
    FP = sum(1 if i == 0 and j == 1 else 0 for i, j in zip(prediction, ground_truth))
    FN = sum(1 if i == 0 and j == 1 else 0 for i, j in zip(prediction, ground_truth))
    TN = sum(1 if i == 0 and j == 0 else 0 for i, j in zip(prediction, ground_truth))
    precision = float(TP)/(FP + TP)
    recall = float(TP)/(FN + TP)
    #print(prediction,ground_truth)
    accuracy = float(sum(prediction==ground_truth))/len(ground_truth)
    f1 = 2*precision*recall/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    return float(sum(prediction == ground_truth))/len(ground_truth)
