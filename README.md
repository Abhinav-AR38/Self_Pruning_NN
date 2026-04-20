# Self-Pruning Neural Network

## Overview
This project implements a self-pruning neural network that learns to remove unnecessary weights during training using learnable gates and L1 regularization.

## Features
- Custom PrunableLinear layer
- Sigmoid-based gating mechanism
- L1 sparsity regularization
- CIFAR-10 dataset training
- Sparsity vs accuracy analysis

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.1    | 49.42%   | 0.65%    |
| 1      | 49.51%   | 3.86%    |
| 5      | 49.01%   | 9.67%    |

## Conclusion
The model successfully demonstrates the trade-off between sparsity and accuracy, showing that higher regularization leads to increased pruning.
