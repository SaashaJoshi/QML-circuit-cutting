# Quantum Machine Learning with Circuit Cutting

Quantum Machine Learning (QML) techniques, including variational Quantum Tensor Networks (QTN), pose a huge implementation challenge regarding qubit requirements. An approach to circumvent this issue is to perform circuit cutting that segments large quantum circuits into multiple smaller sub-circuits that can be trained easily on a quantum device [^1]. This project lays down the workflow for training a variational QTN circuit that implements circuit cutting, specifically gate cutting, to perform data classification. The workflow is built with the help of the Qiskit SDK with dependence on the Circuit Knitting Toolbox [^3] for circuit cutting procedures. Additionally, a significant amount of modifications are made to algorithms like SamplerQNN, a part of the Qiskit Machine Learning package, to accommodate the training of multiple sub-circuits with Qiskit Aer's Sampler runtime primitive. The dataset used for classification is the diabetes dataset from the National Institute of Diabetes and Digestive and Kidney Diseases publicly available on Kaggle.

## Workflow

The training of QML models integrated with circuit cutting can be performed using two possible workflows. 

Workflow A, after performing the circuit cut, trains the sub-circuits with respect to the original train labels. This training happens before the results are combined to reconstruct the expectation value of the original circuit. This procedure helps to maintain a parallel workflow where the sub-circuits can be trained simultaneously. This simultaneous training, over multiple iterations, can be performed with the help of existing Batching techniques in the Qiskit Runtime primitives [^3]. The quasi-probability distributions received at the end of the training and evaluation process are combined to retrieve the expectation values of the original circuit. 

<img src="https://github.com/SaashaJoshi/Cut-QTN/blob/main/graphics/Workflow-A.png" alt="Training Workflow for QML model with Circuit Cutting">

Workflow B, also proposed in [^6], reconstructs the original expectation value after every training iteration. The optimization procedure is performed on the reconstructed expectation value that triggers the model tuning loop. This workflow facilitates simultaneous training over one training iteration at a time. After every training result is received, the sub-circuits go through the classical post-processing steps to reconstruct the original expectation value.

<img src="https://github.com/SaashaJoshi/Cut-QTN/blob/main/graphics/Workflow-B.png" alt="Training Workflow for QML model with Circuit Cutting">



[^1]: D. Guala, S. Zhang, E. Cruz, C. A. Riofrío, J. Klepsch, and J. M. Arrazola, “Practical overview of image classification with tensor-network quantum circuits,” Scientific Reports, vol. 13, no. 1, p. 4427, Mar. 2023, doi: https://doi.org/10.1038/s41598-023-30258-y.
[^2]: Gadi Aleksandrowicz, ‘Qiskit: An Open-source Framework for Quantum Computing’. Zenodo, Jan. 23, 2019. doi: 10.5281/zenodo.2562111.
[^3]: Jim Garrison, ‘Qiskit-Extensions/circuit-knitting-toolbox: Circuit Knitting Toolbox 0.6.0’. Zenodo, Feb. 12, 2024. doi: 10.5281/zenodo.10651875.
[^4]: L. Brenner, C. Piveteau, and D. Sutter, “Optimal wire cutting with classical communication,” arXiv.org, Feb. 07, 2023. https://arxiv.org/abs/2302.03366
[^5]: Mehmet Akturk, “Diabetes Dataset,” Kaggle.com, 2020. https://www.kaggle.com/datasets/mathchi/diabetes-data-set
[^6]: M. Beisel, J. Barzen, M. Bechtold, F. Leymann, F. Truger, and B. Weder, “QuantME4VQA: Modeling and Executing Variational Quantum Algorithms Using Workflows,” Proceedings of the 13th International Conference on Cloud Computing and Services Science, 2023, doi: https://doi.org/10.5220/0011997500003488.
