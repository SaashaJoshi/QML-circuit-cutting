from __future__ import annotations
from typing import Callable
import numpy as np
from qiskit_machine_learning.algorithms import BinaryObjectiveFunction


def create_objective(X: np.ndarray, y: np.ndarray, neural_network, loss):
    return BinaryObjectiveFunction(X, y, neural_network, loss)


def fit(X: np.ndarray, y: np.ndarray):
    function = create_objective(X, y)
    return minimizer(function)


def minimizer(function, initial_point, optimizer: Callable):
    objective = function.objective
    if callable(optimizer):
        optimizer_result = optimizer(
            fun=objective, x0=initial_point, jac=function.gradient,
        )
    else:
        optimizer_result = optimizer.minimize(
            fun=objective,
            x0=initial_point,
            jac=function.gradient,
        )
    return optimizer_result


def print_optimizer_results(optimizer_result):
    print(f"New params (The final point of the minimization): {optimizer_result.x}")
    print(f"The final value of the minimization: {optimizer_result.fun}")
    print(f"The final gradient of the minimization: {optimizer_result.jac}")
    print(f"The total number of function evaluations: {optimizer_result.nfev}")
    print(f"The total number of gradient evaluations: {optimizer_result.njev}")
    print(f"The total number of iterations: {optimizer_result.nit}")


# Objective function may not be required if using this gradient function.
# def gradient(loss, weights, forward_output, weights_grad):
#     loss_gradient = loss.gradient(forward_output, y_train.values.reshape(-1, 1))

#     # for the output we compute a dot product(matmul) of loss gradient for this output
#     # and weights for this output.
#     grad = loss_gradient[:, 0] @ weight_grad[:, 0, :]
#     # we keep the shape of (1, num_weights)
#     grad = grad.reshape(1, -1) / self._num_samples

#     return grad


