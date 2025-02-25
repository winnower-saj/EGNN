# E(n)-Equivariant Graph Neural Network (EGNN)

This repository contains an implementation of the **E(n)-Equivariant Graph Neural Network (EGNN)**, based on the paper:

>**"E(n) Equivariant Graph Neural Networks"**  
> Victor Garcia Satorras, Emiel Hoogeboom, Max Welling.   
> [Paper](https://arxiv.org/abs/2102.09844)

## Overview

EGNN is a graph neural network that maintains **equivariance** to Euclidean transformations (**rotations, translations, and reflections**) while learning representations for graph-structured data. This is particularly useful for tasks in **molecular modeling, physics simulations, and point cloud processing**.

##  Features
- **E(n)-equivariance**: Ensures outputs are consistent under Euclidean transformations.
- **Lightweight architecture**: Avoids using higher-order tensors, making it efficient.
- **Supports node features and edge attributes**.
