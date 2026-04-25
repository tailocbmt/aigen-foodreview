

import torch


def multilabel_accuracy(targets, preds):
    """
    logits:  Tensor of shape [batch_size, num_labels]
    targets: Tensor of shape [batch_size, num_labels], values 0/1
    """
    y_pred = torch.tensor(preds)
    y_true = torch.tensor(targets)
    y_true = y_true.int()

    correct = (y_pred == y_true).float()
    return correct.mean().item()
