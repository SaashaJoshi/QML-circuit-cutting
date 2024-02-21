import numpy as np
from pandas import Series
from qiskit_machine_learning.utils.loss_functions import Loss


class CustomCrossEntropyLoss(Loss):
    r"""
    This class computes the cross entropy loss for each sample as:

    .. math::

        \text{CrossEntropyLoss}(predict, target) = -\sum_{i=0}^{N_{\text{classes}}}
        target_i * log(predict_i).
    """

    def evaluate(self, predict, target):
        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            if isinstance(target, Series):
                target = target.to_numpy().reshape(1, -1)
            else:
                target = target.reshape(1, -1)

        # print("in evaluate, predict and target", predict.shape, target.shape)

        # multiply target and log(predict) matrices row by row and sum up each row
        # into a single float, so the output is of shape(N,), where N number or samples.
        # then reshape
        # before taking the log we clip the predicted probabilities at a small positive number. This
        # ensures that in cases where a class is predicted to have 0 probability we don't get `nan`.
        val = -np.einsum(
            "ij,ij->i", target, np.log2(np.clip(predict, a_min=1e-10, a_max=None))
        ).reshape(-1, 1)

        # val = -np.sum(target * np.log2(np.clip(predict, a_min=1e-10, a_max=None)))
        # print("in evaluate val", val.shape, val)
        return val

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Assume softmax is used, and target vector may or may not be one-hot encoding"""

        self._validate_shapes(predict, target)
        if len(predict.shape) == 1:
            predict = predict.reshape(1, -1)
            if isinstance(target, Series):
                target = target.to_numpy().reshape(1, -1)
            else:
                target = target.reshape(1, -1)

        # sum up target along rows, then multiply predict by this sum element wise,
        # then subtract target
        grad = np.einsum("ij,i->ij", predict, np.sum(target, axis=1)) - target
        print("Gradient in grad", grad)

        return grad