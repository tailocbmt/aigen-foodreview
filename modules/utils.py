

def multilabel_accuracy(targets, preds):
    """
    logits:  Tensor of shape [batch_size, num_labels]
    targets: Tensor of shape [batch_size, num_labels], values 0/1
    """
    targets = targets.int()

    correct = (preds == targets).float()
    return correct.mean().item()
