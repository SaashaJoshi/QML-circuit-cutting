# This code is taken from part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
# Modified by SaashaJoshi.
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
import numpy as np
from numbers import Integral
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSampler, SamplerResult, Sampler
from qiskit_algorithms.gradients import (
    BaseSamplerGradient,
    ParamShiftSamplerGradient,
    SamplerGradientResult,
)
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
import qiskit_machine_learning.optionals as _optionals
logger = logging.getLogger(__name__)


class CustomSampler():
    def __init__(
            self,
            *,
            circuits: tuple[QuantumCircuit],
            sampler=None,
            input_params=None,
            weight_params=None,
            num_subexperiments=None,
            sparse=None,
            interpret=None,
            output_shape=None,
            gradient=None,
            input_gradients=None,
    ):
        # set primitive, provide default
        if sampler is None:
            sampler = Sampler()
        self.sampler = sampler

        # set subexperimets
        if num_subexperiments:
            self.num_subexperiments = num_subexperiments

        # set gradient
        if gradient is None:
            gradient = ParamShiftSamplerGradient(self.sampler)
        self.gradient = gradient

        # Include multiple subexperiement circuits
        self._circuits = circuits
        # self._org_circuit = circuit
        self.subex_circuit = self.circuits[0]

        if all(isinstance(circuit, QNNCircuit) for circuit in self.circuits):
            # Since all the subexperiement circuits will have same inputs and weights
            self._input_params = list(self.subex_circuit.input_parameters)
            self._weight_params = list(self.subex_circuit.weight_parameters)
        else:
            self._input_params = list(input_params) if input_params is not None else []
            self._weight_params = list(weight_params) if weight_params is not None else []

        # if sparse:
        #     _optionals.HAS_SPARSE.require_now("DOK")
        #
        # self.set_interpret(interpret, output_shape)
        # self._input_gradients = input_gradients

        # super().__init__(
        #     num_inputs=len(self._input_params),
        #     num_weights=len(self._weight_params),
        #     sparse=sparse,
        #     output_shape=self._output_shape,
        #     input_gradients=self._input_gradients,
        # )

        # confirm this as well. Or make sure to use measurements beforehand.
        for circuit in self.circuits:
            if len(circuit.clbits) == 0:
                raise ValueError("Subexperiment circuits do not contain clbits.")
        # for circuit in self.circuits:
        #     if len(circuit.clbits) == 0:
        #         circuit = circuit.copy()
        #         circuit.measure_all()

        # check this
        # self._circuit = self._reparameterize_circuit(circuit, input_params, weight_params)

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

    def _forward(self, input_data, weights):
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        # sampler allows batching (Here, batching subexperiments and num_samples)
        # TODO: change this to contain subex circuits
        jobs = []
        for circuit in self._circuits:
            jobs.append(self.sampler.run([circuit] * num_samples, parameter_values))

        # job = self.sampler.run([self._circuits] * num_samples, parameter_values)
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise QiskitMachineLearningError("Sampler job failed.") from exc
        # result = self._postprocess(num_samples, results)

        return results

    def _backward(
            self,
            input_data,
            weights,
    ):
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        input_grad, weights_grad = None, None

        if np.prod(parameter_values.shape) > 0:
            # TODO: change this to contain subex_circuits
            circuits = [self._circuit] * num_samples

            job = None
            if self._input_gradients:
                job = self.gradient.run(circuits, parameter_values)
            elif len(parameter_values[0]) > self._num_inputs:
                params = [self._circuit.parameters[self._num_inputs:]] * num_samples
                job = self.gradient.run(circuits, parameter_values, parameters=params)

            if job is not None:
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError("Sampler job failed.") from exc

                input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        return input_grad, weights_grad
