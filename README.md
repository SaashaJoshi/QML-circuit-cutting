# Quantum Machine Learning with Circuit Cutting

Quantum Machine Learning (QML) techniques, including variational Quantum Tensor Networks (QTN), pose a huge implementation challenge regarding qubit requirements. An approach to circumvent this issue is to perform circuit cutting that segments large quantum circuits into multiple smaller sub-circuits that can be trained easily on a quantum device [^1]. This project lays down the workflow for training a variational QTN circuit that implements circuit cutting, specifically gate cutting, to perform data classification. The workflow is built with the help of the Qiskit SDK with dependence on the Circuit Knitting Toolbox [^3] for circuit cutting procedures. Additionally, a significant amount of modifications are made to algorithms like SamplerQNN, a part of the Qiskit Machine Learning package, to accommodate the training of multiple sub-circuits with Qiskit Aer's Sampler runtime primitive. The dataset used for classification is the diabetes dataset from the National Institute of Diabetes and Digestive and Kidney Diseases publicly available on Kaggle.

## Workflow

The training of QML models integrated with circuit cutting can be performed using two possible workflows.

In workflow A, sub-circuits generated after a circuit cutting undergo training using a subset of input data features. Evaluation is subsequently performed with respect to the original training labels. Following this, the circuits undergo a tuning stage, wherein variable parameters are updated based on the loss function computed in the evaluation stage. Upon obtaining optimal parameter values from the optimizer, the quasi-probability distributions derived from the sub-circuits are combined to reconstruct the expectation value of the original circuit.

This workflow facilitates an independent and parallel evaluation of sub-circuits over multiple iterations. The concurrent training can be performed with the help of existing Batching techniques in the Qiskit Runtime primitives [^3]. 

<img src="https://github.com/SaashaJoshi/Cut-QTN/blob/main/graphics/Workflow-A.png" alt="Training Workflow for QML model with Circuit Cutting">

Workflow B, also proposed in [^6], involves training the sub-circuits to reconstruct the original expectation values after each training iteration. This is unlike Workflow A, where the reconstruction stage occurs after multiple iterations are performed on each sub-circuit. Subsequently, an optimization step is performed on the reconstructed expectation value to tune the variable parameters within the sub-circuits. The updated parameters are then utilized to finalize the training process. The ultimately reconstructed expectation value is further used for validation and testing purposes.

This workflow facilitates concurrent training over one training iteration at a time. This is unlike Workflow A that allows for the parallel execution of sub-circuits over multiple iterations. Following the completion of each training iteration, the sub-circuits undergo classical post-processing steps to reconstruct the original expectation value.

<img src="https://github.com/SaashaJoshi/Cut-QTN/blob/main/graphics/Workflow-B.png" alt="Training Workflow for QML model with Circuit Cutting">

For this project, we have opted to adopt the Workflow A structure. This design decision aligns with considerations related to the constraints imposed by the current Qiskit stack, which currently lacks support for training sub-circuits using the existing SamplerQNN primitive. Additionally, this decision is influenced by the time limitations inherent in the project timeline.

## Evaluation and Results
<!--
<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/all_forward_time.png" alt="Time Taken to Run One Forward Pass on Different Backends.">


<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/all_backward_time.png" alt="Time Taken to Run One Backward Pass on Different Backends.">
-->

<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/final_results_table.png" alt="Results from Training an 8-qubit QML Model with 1 Circuit Cut on CPU and GPU">

<!--
<div class="image-container">
<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/50-iter-cpu/sub-A.png" alt="Training Loss in 4-qubit sub-circuits (A) on a CPU (50 iterations)">
<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/50-iter-cpu/sub-B.png" alt="Training Loss in 4-qubit sub-circuits (B) on a CPU (50 iterations)">
</div>


<div class="image-container">
<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/50-iter-gpu/sub-A.png" alt="Training Loss in 4-qubit sub-circuits (A) on a GPU (50 iterations)">
<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/50-iter-gpu/sub-B.png" alt="Training Loss in 4-qubit sub-circuits (B) on a GPU (50 iterations)">
</div>

<img src="https://github.com/SaashaJoshi/QML-circuit-cutting/blob/main/graphics/CC_time_to_train.png" alt="Time to Train the QML Model with Circuit Cutting on CPU and GPU">
-->


[^1]: D. Guala, S. Zhang, E. Cruz, C. A. Riofrío, J. Klepsch, and J. M. Arrazola, “Practical overview of image classification with tensor-network quantum circuits,” Scientific Reports, vol. 13, no. 1, p. 4427, Mar. 2023, doi: https://doi.org/10.1038/s41598-023-30258-y.
[^2]: Gadi Aleksandrowicz, ‘Qiskit: An Open-source Framework for Quantum Computing’. Zenodo, Jan. 23, 2019. doi: 10.5281/zenodo.2562111.
[^3]: Jim Garrison, ‘Qiskit-Extensions/circuit-knitting-toolbox: Circuit Knitting Toolbox 0.6.0’. Zenodo, Feb. 12, 2024. doi: 10.5281/zenodo.10651875.
[^4]: L. Brenner, C. Piveteau, and D. Sutter, “Optimal wire cutting with classical communication,” arXiv.org, Feb. 07, 2023. https://arxiv.org/abs/2302.03366
[^5]: Mehmet Akturk, “Diabetes Dataset,” Kaggle.com, 2020. https://www.kaggle.com/datasets/mathchi/diabetes-data-set
[^6]: M. Beisel, J. Barzen, M. Bechtold, F. Leymann, F. Truger, and B. Weder, “QuantME4VQA: Modeling and Executing Variational Quantum Algorithms Using Workflows,” Proceedings of the 13th International Conference on Cloud Computing and Services Science, 2023, doi: https://doi.org/10.5220/0011997500003488.
