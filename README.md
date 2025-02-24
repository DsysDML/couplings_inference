# Couplings Inference

[![badge](https://img.shields.io/badge/arXiv-2501.06108%20-red)](https://arxiv.org/abs/2501.06108)

General repository for couplings inference. Code for the Ref. [1].

## Introduction
Understanding complex interactions directly from data is crucial across many disciplines. Many-body interactions shape physics, biology, neuroscience, and social systems, playing a key role in emergence, regulation, and coordination. Although generative models excel at identifying high-order correlations, deriving meaningful insights from them remains challenging. Here, we tackle this problem for generic categorical energy-based generative models and introduce an efficient algorithm to extract high-order couplings from Restricted Boltzmann Machines (RBMs) at affordable times.

<p align="center">
  <img src=https://github.com/DsysDML/couplings_inference/blob/main/figures/pipeline.png?raw=true height="400">
</p>

**Fig. 1. Pipeline of the rbmDCA.**  After the training of the neural network (e.g., an RBM) (b) with data (e.g., MSA) (a), we mapped the trained model (b) onto a Potts-like model (c). Parameters of (c) can be used to predict epistatic contacts in the tertiary structure of the protein. (d) shows the contact prediction obtained for the Response Regulator Receiver Domain family (Pfam entry: PF00072), where light-gray dots are the contacts of the protein, red dots are true positives, and green dots are false positives. We showed the prediction obtained with our RBM-based inference (rbmDCA) in the upper-left part of the matrix, while the prediction obtained with the well-established pseudo-likelihood inference (plmDCA) is shown in the lower-right part. This repo presents how we go from a trained model (b) to the contact prediction (d).

## Contents
- [couplings_inference:](https://github.com/DsysDML/couplings_inference/blob/main/couplings_inference.ipynb) Detailed presentation and implementation of the Python function to compute effective couplings. This implementation requires the PyTorch Library.
- [Data:](https://github.com/DsysDML/couplings_inference/tree/main/data)
    - **PF00072.fasta:** Multiple Sequence Analysis data of the Response Regulator Receiver domain (PF00072) [2]. The original dataset can be found [here](https://github.com/pagnani/ArDCAData).
    - **PF00072_struct.dat:** Structural data for the Response Regulator Receiver domain (PF00072) [2]. The original dataset can be found [here](https://github.com/pagnani/ArDCAData).
    - **plmDCA_score_PF00072.txt:** Contact prediction score with plmDCA used to compare our results. This score was computed using this [repository](https://github.com/pagnani/PlmDCA.jl) [3,4]. 
- [models:](https://github.com/DsysDML/couplings_inference/tree/main/models) A trained RBM model used as example.


## References 
1. Decelle, A., Gómez, A. D. J. N., & Seoane, B. (2025). Inferring High-Order Couplings with Neural Networks. _arXiv preprint arXiv:2501.06108_.
2. Trinquier, J., Uguzzoni, G., Pagnani, A., Zamponi, F., & Weigt, M. (2021). Efficient generative modeling of protein sequences using simple autoregressive models. _Nature communications_, 12(1), 5800.
3. Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E. (2013). Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models. _Physical Review E—Statistical, Nonlinear, and Soft Matter Physics_, 87(1), 012707.
4. Ekeberg, M., Hartonen, T., & Aurell, E. (2014). Fast pseudolikelihood maximization for direct-coupling analysis of protein structure from many homologous amino-acid sequences. _Journal of Computational Physics_, 276, 341-356.
