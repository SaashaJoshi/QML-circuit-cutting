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

from __future__ import annotations
import logging
from typing import cast, Iterable
import numpy as np
from numbers import Integral
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.gradients import (
    BaseSamplerGradient,
    ParamShiftSamplerGradient,
    SamplerGradientResult,
)
import time
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
import qiskit_machine_learning.optionals as _optionals
from primitives.custom_neural_network import CustomNeuralNetwork

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


class PrintHandler(logging.Handler):
    def emit(self, record):
        print(self.format(record))


# Attach the custom handler to the root logger
# logging.getLogger().addHandler(PrintHandler())


class CustomSampler(CustomNeuralNetwork):
    def __init__(
        self,
        *,
        circuits: tuple[QuantumCircuit],
        sampler=None,
        input_params=None,
        weight_params=None,
        sparse=None,
        interpret=None,
        output_shape=None,
        gradient=None,
        input_gradients=False,
    ):
        # set primitive, provide default
        if sampler is None:
            sampler = Sampler()
        self.sampler = sampler

        # set subexperimets
        # self.num_subexperiments = len(circuits)

        # set gradient
        if gradient is None:
            gradient = ParamShiftSamplerGradient(self.sampler)
        self.gradient = gradient

        # Include multiple subexperiement circuits
        # self._circuits = circuits
        # self._org_circuit = circuit
        self.subex_circuit = circuits[0]

        if all(isinstance(circuit, QNNCircuit) for circuit in circuits):
            # Since all the subexperiement circuits will have same inputs and weights
            self._input_params = list(self.subex_circuit.input_parameters)
            self._weight_params = list(self.subex_circuit.weight_parameters)
        else:
            self._input_params = list(input_params) if input_params is not None else []
            self._weight_params = (
                list(weight_params) if weight_params is not None else []
            )

        if sparse:
            _optionals.HAS_SPARSE.require_now("DOK")

        self._sparse = sparse

        self.set_interpret(interpret, output_shape, circuits)
        self._input_gradients = input_gradients
        self._num_inputs = len(self._input_params)
        self._num_weights = len(self._weight_params)

        # self.output_shape = self._output_shape

        CustomNeuralNetwork.__init__(
            self,
            num_inputs=len(self._input_params),
            num_weights=len(self._weight_params),
            sparse=sparse,
            output_shape=self._output_shape,
            input_gradients=self._input_gradients,
        )

        # confirm this as well. Or make sure to use measurements beforehand.
        for circuit in circuits:
            if len(circuit.clbits) == 0:
                raise ValueError("Subexperiment circuits do not contain clbits.")
        # for circuit in self.circuits:
        #     if len(circuit.clbits) == 0:
        #         circuit = circuit.copy()
        #         circuit.measure_all()

        # check this
        self._circuits = [
            self._reparameterize_circuit(circuit, input_params, weight_params)
            for circuit in circuits
        ]

    @property
    def circuits(self):
        """Returns the underlying quantum circuit."""
        # return self._org_circuit
        return self._circuits

    @property
    def input_params(self):
        """Returns the list of input parameters."""
        return self._input_params

    @property
    def weight_params(self):
        """Returns the list of trainable weights parameters."""
        return self._weight_params

    # COMPLETED
    def set_interpret(
        self,
        interpret=None,
        output_shape=None,
        circuits=None,
    ) -> None:
        # derive target values to be used in computations
        # self._output_shape = (
        #     len(circuits),
        #     2 ** self.subex_circuit.num_qubits,
        # )
        self._output_shape = self._compute_output_shape(interpret, output_shape)
        self._interpret = interpret if interpret is not None else lambda x: x

    def _compute_output_shape(
        self,
        interpret=None,
        output_shape=None,
    ) -> tuple[int, ...]:
        """Validate and compute the output shape."""

        # this definition is required by mypy
        output_shape_: tuple[int, ...] = (-1,)

        if interpret is not None:
            if output_shape is None:
                raise QiskitMachineLearningError(
                    "No output shape given; it's required when using custom interpret!"
                )
            if isinstance(output_shape, Integral):
                output_shape = int(output_shape)
                output_shape_ = (output_shape,)
            else:
                output_shape_ = output_shape  # type: ignore
        else:
            if output_shape is not None:
                # Warn user that output_shape parameter will be ignored
                logger.warning(
                    "No interpret function given, output_shape will be automatically "
                    "determined as 2^num_qubits."
                )
            output_shape_ = (2**self.subex_circuit.num_qubits,)

        # (16, )
        # print(output_shape_)

        return output_shape_

    # COMPLETED
    def _preprocess_forward(self, input_data, weights):
        if input_data is not None:
            num_samples = input_data.shape[0]
            if weights is not None:
                weights = np.broadcast_to(weights, (num_samples, len(weights)))
                parameters = np.concatenate((input_data, weights), axis=1)
            else:
                parameters = input_data
        else:
            if weights is not None:
                num_samples = 1
                parameters = np.broadcast_to(weights, (num_samples, len(weights)))
            else:
                # no input, no weights, just execute circuit once
                num_samples = 1
                parameters = np.asarray([])

        # Increase num_params to include subexperiments
        # parameters = [[params] * self.num_subexperiments for params in parameters]
        # print("params", len(parameters), len(parameters[0]), len(parameters[0][0]))
        return parameters, num_samples

    # COMPLETED
    def _postprocess(self, num_samples, results):
        """
        Post-processing during forward pass of the network.
        """
        # results = 6, (537)

        if self._sparse:
            # pylint: disable=import-error
            from sparse import DOK

            prob = {
                num_sample: DOK((num_samples, *self._output_shape))
                for num_sample in range(len(results))
            }
        else:
            prob = {
                num_sample: np.zeros((num_samples, *self._output_shape))
                for num_sample in range(len(results))
            }

        # probs = 6, (537, 16)

        # 6, (537)
        dict_of_list_of_quasi_dists = {
            index: result.quasi_dists for index, result in results.items()
        }
        # print(dict_of_list_of_quasi_dists)
        # 6 keys, 537 quasi_dists (dict_of_counts) per keys
        # print(len(dict_of_quasi_dists), len(dict_of_quasi_dists[0]))

        # for i in range(num_samples):
        # i = 1
        # evaluate probabilities

        # CHECK FROM HERE
        # for i_num_subex, list_of_counts in dict_of_list_of_quasi_dists.items():
        #     # should be 6, 537 # wrong it is 6, 16
        #     # print(len(dict_of_list_of_quasi_dists), len(list_of_counts))
        #     for i_num_sample, counts in enumerate(list_of_counts):
        #         for b, v in counts.items():
        #             key = self._interpret(b)
        #             if isinstance(key, Integral):
        #                 key = (cast(int, key),)
        #             key = (i_num_sample, i_num_subex, *key)  # type: ignore
        #             # print(key)
        #             prob[key] += v

        for i_num_subex, list_of_counts in dict_of_list_of_quasi_dists.items():
            for i, counts in enumerate(list_of_counts):
                # 6, 537, 2 or more
                # print(len(dict_of_list_of_quasi_dists), len(list_of_counts), len(counts))
                # print(counts)
                for b, v in counts.items():
                    key = self._interpret(b)
                    if isinstance(key, Integral):
                        key = (cast(int, key),)
                        # print(key)
                    key = (i, *key)  # type: ignore
                    # print(key)
                    prob[i_num_subex][key] += v

        # should be 6 (537, 16)
        # print(len(prob), prob[0].shape)
        # prob = np.array(prob)
        if self._sparse:
            return prob.to_coo()
        else:
            return prob

    def _postprocess_gradient(self, num_samples, results):
        """
        Post-processing during backward pass of the network.
        """

        if self._sparse:
            # pylint: disable=import-error
            from sparse import DOK

            # shape = (537, 16, 4)
            input_grad = {
                num_sample: (
                    DOK((num_samples, *self._output_shape, self._num_inputs))
                    if self._input_gradients
                    else None
                )
                for num_sample in range(len(results))
            }
            # shape = (537, 16, 7)
            weights_grad = {
                num_sample: DOK((num_samples, *self._output_shape, self._num_weights))
                for num_sample in range(len(results))
            }
        else:

            input_grad = {
                num_sample: (
                    np.zeros((num_samples, *self._output_shape, self._num_inputs))
                    if self._input_gradients
                    else None
                )
                for num_sample in range(len(results))
            }
            weights_grad = {
                num_sample: np.zeros(
                    (num_samples, *self._output_shape, self._num_weights)
                )
                for num_sample in range(len(results))
            }

        # num_param_gradients = num of parameters per circuit
        # For example, 7 for sub_circuit["A"]
        if self._input_gradients:
            num_param_gradients = self._num_inputs + self._num_weights
        else:
            num_param_gradients = self._num_weights

        dict_of_list_of_list_of_gradients = {
            index: result.gradients for index, result in results.items()
        }
        # 6 keys, 537 gradient sample lists (list_of_list_of_grads) per keys, 7 gradient dicts per sample
        # print(len(dict_of_list_of_list_of_gradients))

        for (
            i_num_subex,
            list_of_list_of_grads,
        ) in dict_of_list_of_list_of_gradients.items():
            for i_num_sample, list_of_grads in enumerate(
                list_of_list_of_grads[:num_param_gradients]
            ):
                for i_num_param, grads in enumerate(list_of_grads):
                    # 537, 11, 2
                    # print(len(list_of_list_of_grads), len(list_of_grads), len(grads))
                    for k, val in grads.items():
                        # get index for input or weights gradients
                        if self._input_gradients:
                            grad_index = (
                                i_num_sample
                                if i_num_sample < self._num_inputs
                                else i_num_sample - self._num_inputs
                            )
                            # print(grad_index)
                        else:
                            # grad_index = i_num_param
                            grad_index = i_num_sample
                            # print(grad_index)

                        # interpret integer and construct key
                        key = self._interpret(k)
                        # print(key)
                        if isinstance(key, Integral):
                            # shape =
                            key = (
                                i_num_sample,
                                int(key),
                                grad_index,
                            )
                        else:
                            # if key is an array-type, cast to hashable tuple
                            key = tuple(cast(Iterable[int], key))
                            key = (i_num_sample, *key, grad_index)

                        # store value for inputs or weights gradients
                        # print(key)
                        if self._input_gradients:
                            # we compute input gradients first
                            if i_num_param < self._num_inputs:
                                input_grad[i_num_subex][key] += val
                            else:
                                weights_grad[i_num_subex][key] += val
                        else:
                            weights_grad[i_num_subex][key] += val

        if self._sparse:
            if self._input_gradients:
                input_grad = input_grad.to_coo()  # pylint: disable=no-member
            weights_grad = weights_grad.to_coo()

        # print(len(weights_grad))

        return input_grad, weights_grad

    # COMPLETED
    def _forward(self, input_data, weights):
        # print("Forward called by some function.")
        logging.info("Forward is being called.")
        start_time = time.time()

        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        # sampler allows batching (Here, batching subexperiments and num_samples)
        # This contains subex circuits.
        jobs = {
            index: self.sampler.run([circuit] * num_samples, parameter_values)
            for index, circuit in enumerate(self._circuits)
        }

        try:
            # Dictionary of SamplerResult dictionaries
            results = {index: job.result() for index, job in jobs.items()}
        except Exception as exc:
            raise QiskitMachineLearningError("Sampler job failed.") from exc

        # 6, 537
        # print("forward results", len(results), len(results[0].quasi_dists))
        result = self._postprocess(num_samples, results)

        # 6, (537, 16)
        # print(len(result), result[0].shape)
        # return result, results

        end_time = time.time()
        running_time = end_time - start_time
        print("Running Time for Forward Pass: %s seconds", running_time)
        logger.info("Running Time for Forward Pass: %s seconds", running_time)

        return result

    def _backward(
        self,
        input_data,
        weights,
    ):
        """Backward pass of the network."""
        # print("Backward called by some function.")
        logging.info("Backward is being called.")
        start_time = time.time()
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        input_grad, weights_grad = None, None

        if np.prod(parameter_values.shape) > 0:
            jobs = None
            if self._input_gradients:
                jobs = {
                    index: self.gradient.run([circuit] * num_samples, parameter_values)
                    for index, circuit in enumerate(self._circuits)
                }
            elif len(parameter_values[0]) > self._num_inputs:
                params = [
                    self.subex_circuit.parameters[self._num_inputs :]
                ] * num_samples
                jobs = {
                    index: self.gradient.run([circuit] * num_samples, parameter_values)
                    for index, circuit in enumerate(self._circuits)
                }
                # jobs = {
                #     index: self.gradient.run(
                #         [circuit] * num_samples, parameter_values, parameters=params
                #     )
                #     for index, circuit in enumerate(self._circuits)
                # }

            if jobs is not None:
                try:
                    results = {index: job.result() for index, job in jobs.items()}
                except Exception as exc:
                    raise QiskitMachineLearningError("Sampler job failed.") from exc

                input_grad, weights_grad = self._postprocess_gradient(
                    num_samples, results
                )

        # return input_grad, weights_grad, results

        end_time = time.time()
        running_time = end_time - start_time
        print("Running Time for Backward Pass: %s seconds", running_time)
        logger.info("Running Time for Backward Pass: %s seconds", running_time)

        return input_grad, weights_grad
