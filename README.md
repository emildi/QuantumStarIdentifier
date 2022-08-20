# Quantum Star Identifier
![QSI](QSI.jpg?raw=true "QSI")

## Introduction

This repository contains the Python code used to produce the quantum SVM results in the MSc in AI dissertation with the title `"Investigate the Feasibility of
Using a Quantum-Enhanced SVM Classifier for Star Identification"`. The dataset used is nto public, as was extracted from the private dataset (which was in turn generated from the Hipparcos catalog) created for the article [Efficient Star Identification Using a Neural Network](https://www.mdpi.com/1424-8220/20/13/3684) by David Rijlaarsdam et al.
<br><br>
Based on ideas from Qiskit's tutorial on [Quantum Kernel Machine Learning](https://github.com/Qiskit/qiskit-machine-learning/blob/main/docs/tutorials/03_quantum_kernel.ipynb).

## Software Environment

### Prerequisites

The software development and experiments were done on a Linux workstation (Ubuntu 20.04, CUDA 11.4) or HPC nodes (CentOS 7, CUDA 10.1.243). To simulate large number of qubits and/or deep quantum circuits using a modern NVidia GPU and abundant amount of RAM is highly recommended.

Instructions on how to build a similar conda environment (which should work on number of different platforms, like macOS or Windows) or recreate exact replica on the conda environment (this would only work on Linux x86-64) are in the following sections (this assumes conda and pip are already installed and configured).

## Build a similar conda environment

Run the following command, which would create and activate a new conda environment called `qsi` (the name could be changed as desired) with the required packages:

```bash
conda create -n qsi python=3.8 -y
conda activate qsi
```

Install the required packages:

```bash
pip install qiskit==0.34.2
pip install qiskit-machine-learning==0.3.1

conda install -c conda-forge pandas  -y

conda install jupyter notebook -y
conda install -c conda-forge matplotlib-base -y
```

Start a Jupyter notebook or Jupyter Lab instance and access the notebook from it.

## Build a replica conda environment (Linux x86-64 only)

To build an exact replica of the conda software environment on a Linux x86-64 machine run the following command, which would create a new conda environment called `qsi` (the name could be changed as desired) with the required packages:

```bash
conda create --name qsi --file spec-file.txt
```

Activate the conda environment and run the python file `qsi-qsvm-qiskit-final.py`:

```bash
conda activate qsi
python qsi-qsvm-qiskit-final.py
```
> **_NOTE:_**  Simulating quantum circuits at scale requires significant classical hardware re- sources. Sometimes those simulations could take multiple hours and days and require substantial computational and memory resources. The HPC nodes used to run the experiments had the following configuration: 2x 20-core 2.4 GHz Intel Xeon Gold 6148 (Skylake) processors, 192 GiB of RAM, 2x NVIDIA Tesla V100 16GB PCIe (Volta architecture) GPUs.