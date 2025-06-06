import torch
import numpy as np
from dataUtils import data

# credit: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
def eval(
    prediction, 
    truth, 
    threshold=0.5
):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    prediction = (prediction>=threshold).int()
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float('inf')).item()
    FN = torch.sum(confusion_vector == 0).item()
    BATCH_MACRO = torch.stack((torch.sum(confusion_vector == 1, dim=0),
                               torch.sum(confusion_vector == float('inf'), dim=0),
                               torch.sum(confusion_vector == 0, dim=0)))

    # Jaccard metric
    jaccard = prediction.numpy()
    INDICES = set(np.where(jaccard==1)[1])
    return TP, FP, FN, BATCH_MACRO, INDICES

def evalCalc(
    EPOCH_TP, 
    EPOCH_FP, 
    EPOCH_FN, 
    EPOCH_MACRO, 
    JACCARD, 
    TYPE
):
    """
    Calculate the micro and macro precision, recall, f1 metrics
    """
    evaluation = {}
    # Metric calc - micro
    if (EPOCH_FP+EPOCH_TP==0) or (EPOCH_FN+EPOCH_TP==0) or (EPOCH_TP==0):
        # handles edge case when model predicts everything negative
        precision_micro = "N/A"
        recall_micro    = "N/A"
        F1_micro        = "N/A"
    else:
        precision_micro = EPOCH_TP/(EPOCH_FP+EPOCH_TP)
        recall_micro    = EPOCH_TP/(EPOCH_FN+EPOCH_TP)
        F1_micro        = 2*precision_micro*recall_micro/(precision_micro+recall_micro)

    # Metric calc - macro
    precision = torch.div(EPOCH_MACRO[0], EPOCH_MACRO[0]+EPOCH_MACRO[1])
    precision[torch.isnan(precision)] = 0
    recall = torch.div(EPOCH_MACRO[0], EPOCH_MACRO[0]+EPOCH_MACRO[2])
    recall[torch.isnan(recall)] = 0
    if (torch.sum(precision)==0) or (torch.sum(recall)==0) or (torch.sum(EPOCH_MACRO[0])==0):
        # handles edge case when model predicts everything negative
        precision_macro = "N/A"
        recall_macro    = "N/A"
        F1_macro        = "N/A"
    else:
        precision_macro = torch.mean(precision).item()
        recall_macro    = torch.mean(recall).item()
        F1_macro        = 2*precision_macro*recall_macro/(precision_macro+recall_macro)

    # Jaccard metric
    if (TYPE == "training"):
        j_metric = float(len(JACCARD & set(data.training_index))) / float(len(JACCARD | set(data.training_index)))
    elif (TYPE == "validation"):
        j_metric = float(len(JACCARD & set(data.validation_index))) / float(len(JACCARD | set(data.validation_index)))
    elif (TYPE == "testing"):
        j_metric = float(len(JACCARD & set(data.testing_index))) / float(len(JACCARD | set(data.testing_index)))

    evaluation["precision_micro"] = precision_micro
    evaluation["recall_micro"]    = recall_micro
    evaluation["F1_micro"]        = F1_micro
    evaluation["precision_macro"] = precision_macro
    evaluation["recall_macro"]    = recall_macro
    evaluation["F1_macro"]        = F1_macro
    evaluation["Jaccard"]         = j_metric
    return evaluation