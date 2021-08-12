# Deep Learning Drug Discovery

This repository holds the code for drug-target interaction prediction using deep learning.

## Installation

The packages required for this code to work are:

* torch
* torch_geometric
* pytorch_lightning

## Data creation

`snakemake` is used to create the data from protein structures.

The pipeline works as follows:

1. Put all the protein structures in the `resources/structures` directory
1. From the root directory run the `snakemake -j 4` pipeline (this will run it with 4 cores)
1. In the end you should obtain a file in the `results/prepare_proteins` folder which will contain a dataframe with calculated graph data for each structure (in a pickle format)