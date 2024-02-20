# This code is taken from part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
# Modified by SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import copy
from qiskit_machine_learning.algorithms import ObjectiveFunction


def generate_empty_dict(num):
    empty_dict = {}
    for num_samples in range(num):
        empty_dict[num_samples] = []
    return empty_dict


class CustomMultiClassObjectiveFunction(ObjectiveFunction):
    """
    An objective function for multiclass representation of the output. For instance, classes of
    ``0``, ``1``, ``2``, etc.
    """

    # Objective function for the 0th sub-circuits
    def objective0(self, weights):
        # probabilities is of shape (N, num_outputs)
        # shape 6, (537, 16)
        probs = self._neural_network_forward(weights)
        # print(len(probs), probs[0].shape)

        num_outputs = self._neural_network.output_shape[0]  # 16
        val = 0.0
        num_samples = self._X.shape[0]  # 537

        # val_dict = copy.deepcopy(generate_empty_dict(len(probs)))
        val_list = np.zeros((6,), dtype=float)
        # for num_subcirc in range(len(probs)):
        for i in range(num_outputs):
            # for each output we compute a dot product of probabilities of this output and a loss
            # vector.
            # loss vector is a loss of a particular output value(value of i) versus true labels.
            # we do this across all samples.
            val += probs[0][:, i] @ self._loss(np.full(num_samples, i), self._y)
            # val_list[num_subcirc] = val / self._num_samples
        val = val / self._num_samples

        # print(val_list)
        # return val_list
        return val

    # Objective function for the 1st sub-circuits
    def objective1(self, weights):
        # probabilities is of shape (N, num_outputs)
        # shape 6, (537, 16)
        probs = self._neural_network_forward(weights)
        num_outputs = self._neural_network.output_shape[0]  # 16
        val = 0.0
        num_samples = self._X.shape[0]  # 537

        # for num_subcirc in range(len(probs)):
        for i in range(num_outputs):
            val += probs[1][:, i] @ self._loss(np.full(num_samples, i), self._y)
        val = val / self._num_samples

        return val

    # Objective function for the 2nd sub-circuits
    def objective2(self, weights):
        # probabilities is of shape (N, num_outputs)
        # shape 6, (537, 16)
        probs = self._neural_network_forward(weights)
        num_outputs = self._neural_network.output_shape[0]  # 16
        val = 0.0
        num_samples = self._X.shape[0]  # 537

        # for num_subcirc in range(len(probs)):
        for i in range(num_outputs):
            val += probs[2][:, i] @ self._loss(np.full(num_samples, i), self._y)
        val = val / self._num_samples

        return val

    # Objective function for the 3rd sub-circuits
    def objective3(self, weights):
        # probabilities is of shape (N, num_outputs)
        # shape 6, (537, 16)
        probs = self._neural_network_forward(weights)
        num_outputs = self._neural_network.output_shape[0]  # 16
        val = 0.0
        num_samples = self._X.shape[0]  # 537

        # for num_subcirc in range(len(probs)):
        for i in range(num_outputs):
            val += probs[3][:, i] @ self._loss(np.full(num_samples, i), self._y)
        val = val / self._num_samples

        return val

    # Objective function for the 04th sub-circuits
    def objective4(self, weights):
        # probabilities is of shape (N, num_outputs)
        # shape 6, (537, 16)
        probs = self._neural_network_forward(weights)
        num_outputs = self._neural_network.output_shape[0]  # 16
        val = 0.0
        num_samples = self._X.shape[0]  # 537

        # for num_subcirc in range(len(probs)):
        for i in range(num_outputs):
            val += probs[4][:, i] @ self._loss(np.full(num_samples, i), self._y)
        val = val / self._num_samples

        return val

    # Objective function for the 5th sub-circuits
    def objective5(self, weights):
        # probabilities is of shape (N, num_outputs)
        # shape 6, (537, 16)
        probs = self._neural_network_forward(weights)
        num_outputs = self._neural_network.output_shape[0]  # 16
        val = 0.0
        num_samples = self._X.shape[0]  # 537

        # for num_subcirc in range(len(probs)):
        for i in range(num_outputs):
            val += probs[5][:, i] @ self._loss(np.full(num_samples, i), self._y)
        val = val / self._num_samples

        return val

    # Gradient for 0th sub-circuits
    def gradient0(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        # shape = 6, (537, 16, 7)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
        # grad_list = []
        num_samples = self._X.shape[0]  # 537
        num_outputs = self._neural_network.output_shape[0]  # 16

        # for num_subcirc in range(len(weight_prob_grad)):
        for i in range(num_outputs):
            # similar to what is in the objective, but we compute a matrix multiplication of
            # weight probability gradients and a loss vector.
            grad += weight_prob_grad[0][:, i, :].T @ self._loss(
                np.full(num_samples, i), self._y
            )
        # grad_list.append(grad / self._num_samples)
        grad = grad / self._num_samples

        # print(grad_list)
        return grad

    # Gradient for 1st sub-circuits
    def gradient1(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        # shape = 6, (537, 16, 7)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
        num_samples = self._X.shape[0]  # 537
        num_outputs = self._neural_network.output_shape[0]  # 16

        for i in range(num_outputs):
            grad += weight_prob_grad[1][:, i, :].T @ self._loss(
                np.full(num_samples, i), self._y
            )
        grad = grad / self._num_samples

        return grad

    # Gradient for 2nd sub-circuits
    def gradient2(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        # shape = 6, (537, 16, 7)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
        num_samples = self._X.shape[0]  # 537
        num_outputs = self._neural_network.output_shape[0]  # 16

        for i in range(num_outputs):
            grad += weight_prob_grad[2][:, i, :].T @ self._loss(
                np.full(num_samples, i), self._y
            )
        grad = grad / self._num_samples

        return grad

    # Gradient for 3rd sub-circuits
    def gradient3(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        # shape = 6, (537, 16, 7)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
        num_samples = self._X.shape[0]  # 537
        num_outputs = self._neural_network.output_shape[0]  # 16

        for i in range(num_outputs):
            grad += weight_prob_grad[3][:, i, :].T @ self._loss(
                np.full(num_samples, i), self._y
            )
        grad = grad / self._num_samples

        return grad

    # Gradient for 4th sub-circuits
    def gradient4(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        # shape = 6, (537, 16, 7)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
        num_samples = self._X.shape[0]  # 537
        num_outputs = self._neural_network.output_shape[0]  # 16

        for i in range(num_outputs):
            grad += weight_prob_grad[4][:, i, :].T @ self._loss(
                np.full(num_samples, i), self._y
            )
        grad = grad / self._num_samples

        return grad

    # Gradient for 5th sub-circuits
    def gradient5(self, weights: np.ndarray) -> np.ndarray:
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        # shape = 6, (537, 16, 7)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
        num_samples = self._X.shape[0]  # 537
        num_outputs = self._neural_network.output_shape[0]  # 16

        for i in range(num_outputs):
            grad += weight_prob_grad[5][:, i, :].T @ self._loss(
                np.full(num_samples, i), self._y
            )
        grad = grad / self._num_samples

        return grad

    # def gradient(self, weights: np.ndarray) -> np.ndarray:
    #     # weight probability gradient is of shape (N, num_outputs, num_weights)
    #     # shape = 6, (537, 16, 7)
    #     _, weight_prob_grad = self._neural_network.backward(self._X, weights)
    #
    #     grad = np.zeros((1, self._neural_network.num_weights))  # Shape = (1, 7)
    #     grad_list = []
    #     num_samples = self._X.shape[0]  # 537
    #     num_outputs = self._neural_network.output_shape[0]  # 16
    #
    #     for num_subcirc in range(len(weight_prob_grad)):
    #         for i in range(num_outputs):
    #             # similar to what is in the objective, but we compute a matrix multiplication of
    #             # weight probability gradients and a loss vector.
    #             grad += weight_prob_grad[num_subcirc][:, i, :].T @ self._loss(np.full(num_samples, i), self._y)
    #         grad_list.append(grad / self._num_samples)
    #
    #     # print(grad_list)
    #     return grad_list


# class CustomBinaryObjectiveFunction(ObjectiveFunction):
#     """An objective function for binary representation of the output. For instance, classes of
#     ``-1`` and ``+1``."""

#     def objective(self, weights: np.ndarray) -> float:
#         # predict is of shape (N, 1), where N is a number of samples
#         # predict_shape = self._neural_network_forward(weights)

#         # shape 537, 6, 16
#         predicts = self._neural_network_forward(weights)[0]
#         targets = np.array(self._y).reshape(predicts.shape)

#         # float(...) is for mypy compliance

#         objectives = []
#         for (predict, target) in zip(predicts[0], targets[0]):
#             # print(len(predict), len(target))
#             objectives.append(float(np.sum(self._loss(predict, target)) / self._num_samples))

#         return objectives

#     def gradient(self, weights: np.ndarray) -> np.ndarray:
#         # check that we have supported output shape
#         num_outputs = self._neural_network.output_shape
#         print(num_outputs) # (6, 16)

#         # if num_outputs != 1:
#         #     raise ValueError(f"Number of outputs is expected to be 1, got {num_outputs}")

#         # output must be of shape (N, 1), where N is a number of samples
#         output = self._neural_network_forward(weights)
#         print(output.shape) # (537, 6, 16)

#         # weight grad is of shape (N, 1, num_weights)
#         _, weight_grad = self._neural_network.backward(self._X, weights)
#         print(weight_grad.shape) # (537, 6, 16, 7)

#         # we reshape _y since the output has the shape (N, 1) and _y has (N,)
#         # loss_gradient is of shape (N, 1)
#         # Loss gradient should now be of size (537, 6) for 6 different subex-circs
#         self._y = np.array(self._y).reshape(output.shape)

#         empty_dict = {}
#         for num_samples in range(537):
#             empty_dict[num_samples] = []

#         loss_gradient = copy.deepcopy(empty_dict)
#         for num_samples in range(6):
#             for (sub_y, sub_output) in zip(self._y[0][num_samples], output[0][num_samples]):
#                 print(sub_y, sub_output)
#                 loss_gradient[num_samples].append(self._loss.gradient(sub_output, sub_y))
#         print(loss_gradient, len(loss_gradient))

#         # for the output we compute a dot product(matmul) of loss gradient for this output
#         # and weights for this output.
#         # print(loss_gradient[0][:, 0], len(loss_gradient[0][:, 0]))
#         # print(weight_grad[:, 0, :], len(weight_grad[:, 0, :]))

#         grad = loss_gradient[:, 0] @ weight_grad[:, 0, :]
#         print(len(grad), len(grad[0]), grad.shape)

#         # we keep the shape of (1, num_weights)
#         grad = grad.reshape(1, -1) / self._num_samples


#         return grad
