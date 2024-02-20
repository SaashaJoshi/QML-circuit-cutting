import copy
import numpy as np
from qiskit_aer.primitives import Sampler
from qiskit.primitives import SamplerResult
from circuit_knitting.cutting import reconstruct_expectation_values


def generate_empty_dict(num):
    empty_dict = {}
    for num_samples in range(num):
        empty_dict[num_samples] = []
    return empty_dict


def get_reconstructed_expvals(A_dict, B_dict, coefficients, sub_observables):
    zip_dict = zip(A_dict.items(), B_dict.items())
    reconstructed_expvals = []

    for (num_subex1, v1), (num_subex2, v2) in zip_dict:
        # print((num_subex1, v1), (num_subex2, v2))
        # combine_dict = {"A": None, "B": None}
        # combine_dict["A"] = v1
        # combine_dict["B"] = v2
        combine_dict = {"A": v1, "B": v2}
        # print(len(combine_dict["A"].quasi_dists))
        # print(combine_dict["A"].quasi_dists)
        # print(v1.quasi_dists)
        reconstructed_expvals.append(
            reconstruct_expectation_values(combine_dict, coefficients, sub_observables)[
                0
            ]
        )

    return reconstructed_expvals


def get_subcircuit_results(x_test, final_circuits, optimizer_results):
    sampler = Sampler()
    test_results = copy.deepcopy(generate_empty_dict(len(x_test.values)))
    for num_sample, x_val in enumerate(x_test.values):
        for num_subex, (subex, result) in enumerate(
            zip(final_circuits, optimizer_results)
        ):
            param = np.append(x_val, result)
            # param = np.append(x_val, result.x)    Use this. Keeping above temporarily to avoid retraining.
            circ = subex.assign_parameters(param, inplace=False)
            test_results[num_sample].append(sampler.run(circ).result())

    return test_results


def get_dict_sampler_results(x_test, subexperiments, test_results):
    dists_dict = copy.deepcopy(generate_empty_dict(len(x_test.values)))
    metadata_dict = copy.deepcopy(generate_empty_dict(len(x_test.values)))

    for num_samples in range(len(x_test.values)):
        for num_subex in range(len(subexperiments)):
            dists_dict[num_samples].append(
                *test_results[num_samples][num_subex].quasi_dists
            )
            metadata_dict[num_samples].append(
                *test_results[num_samples][num_subex].metadata
            )

    dict_sampler_results = copy.deepcopy(generate_empty_dict(len(x_test.values)))
    for num_samples, _ in enumerate(x_test.values):
        dict_sampler_results[num_samples] = SamplerResult(
            quasi_dists=dists_dict[num_samples], metadata=metadata_dict[num_samples]
        )

    return dict_sampler_results
