# Kähler Quantisation for Molecular Surface Approximation (KQMolSA)

## Introduction

Ligand-based virtual screening aims to reduce the cost and duration of drug discovery campaigns. Shape similarity can be used to screen large databases, with the goal of predicting potential new hits by comparing to 
molecules with known favourable properties. KQMolSA is a new alignment-free surface-based molecular shape descriptor derived from the theory 
of Kähler Quantization, a subfield of Riemannian geometry. The shape of the molecule is approximated by a 
(2k+1)x(2k+1) _Hermitian matrix_. The full method is described [here](https://arxiv.org/pdf/2301.04424.pdf).


## In Development

KQMolSA should currently be considered a beta version under development. This initial sample runs for the supplied PDE5 inhibitor test sets (as discussed in the above paper), but is not guaranteed to work for all molecules.

## Installation

#### Dependencies
- Local Installation of [Anaconda](https://www.anaconda.com)
- [RDKit](https://www.rdkit.org/docs/Install.html)

#### Downloading the Software
Run the following in the terminal from the directory the software is to be cloned to:
```
git clone https://github.com/RPirie96/KQMolSA.git
```

Create a conda environment for the required dependencies (note this was created for MacOS but should work for other OS too)
```
conda env create -f environment.yml
```

For Windows machines use:
```
conda env create -f environment_windows.yml
```

## Running KQMolSA

The Jupyter Notebook "run_KQMolSA.ipynb" can be used to run the code for the examples provided in the paper. Note that you'll need to change the paths specified in the 1st cell to the directory the python scripts and data have been cloned to for the notebook to run.

The script "example_run.py" is a script to run the method for any dataset. It takes in 3 arguments:
- name of the file containing molecule structures
- name of the file to write the produced conformers to
- name of the csv file to write the final set of scores to

#### Data Supplied
- SVT.sdf: structure data file containing a single conformer for each of Sildenafil, Vardenafil and Tadalafil.
- X_confs_10.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 low energy conformers generated using RDKit. 
- X_confs_10random.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 random conformers generated using RDKit.
