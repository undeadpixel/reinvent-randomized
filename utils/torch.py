"""
PyTorch related util functions
"""
import torch

def set_default_device(device_name):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if device_name == "cpu":
        tensor = torch.FloatTensor
    elif device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
    torch.set_default_tensor_type(tensor)


# TODO: change that for pytorch function
def nl_loss(inputs, targets):
    """
    Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

    :param inputs: (batch_size, num_classes) *Log probabilities of each class*.
    :param targets: (batch_size) *Target class index*.
    :return: loss : (batch_size) *Loss for each example*.
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = target_expanded * inputs  # FIXME: do we need that?
    loss = torch.sum(loss, 1)
    return loss
