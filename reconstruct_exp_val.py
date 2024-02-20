import copy
import numpy as np
from qiskit_aer.primitives import Sampler
from qiskit.primitives import SamplerResult


def generate_empty_dict(num):
    empty_dict = {}
    for num_samples in range(num):
        empty_dict[num_samples] = []
    return empty_dict


def custom_reconstruct():
    pass


def get_subcircuit_results(x_test, final_circuits, optimizer_results):
    sampler = Sampler()
    test_results = copy.deepcopy(generate_empty_dict(len(x_test.values)))
    for num_sample, x_val in enumerate(x_test.values):
        for num_subex, (subex, result) in enumerate(zip(final_circuits, optimizer_results)):
            param = np.append(x_val, result.x)
            circ = subex.assign_parameters(param, inplace=False)
            test_results[num_sample].append(sampler.run(circ).result())

    return test_results


def get_dict_sampler_results(x_test, subexperiments, test_results):
    dists_dict = copy.deepcopy(generate_empty_dict(len(x_test.values)))
    metadata_dict = copy.deepcopy(generate_empty_dict(len(x_test.values)))

    for num_samples in range(len(x_test.values)):
        for num_subex in range(len(subexperiments)):
            dists_dict[num_samples].append(*test_results[num_samples][num_subex].quasi_dists)
            metadata_dict[num_samples].append(*test_results[num_samples][num_subex].metadata)

    dict_sampler_results = copy.deepcopy(generate_empty_dict(len(x_test.values)))
    for index, _ in enumerate(subexperiments):
        dict_sampler_results[index] = [SamplerResult(quasi_dists=dists_dict[index], metadata=metadata_dict[index])]

    return dict_sampler_results


def get_combined_dict(A_dict, B_dict):
    combine_dict = {"A": None, "B": None}
    zip_dict = zip(A_dict.items(), B_dict.items())

    for (num_subex1, v1), (num_subex2, v2) in zip_dict:
        # print((num_subex1, v1), (num_subex2, v2))
        # combine_dict = {"A": None, "B": None}
        combine_dict["A"] = v1
        combine_dict["B"] = v2

    return combine_dict