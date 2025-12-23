# Analysis: SGD Optimizer Experiment

## Overview
This document summarizes the optimization experiments conducted using SGD on the Dungeon Soup Data.

## Specifications
- **Model**: Convolutional Neural Network (CNN) 
- **Optimizer**: SGD
- **Epochs**: 35 (with Early Stopping, patience=5)
- **Batch Size**: 32
- **Tested Learning Rates (LR)**: `0.001`, `0.01`, `0.05`

## Findings (58 Classes)

### 1. Learning Rate = 0.001 (Underfitting)
- **Behavior**: Extremely slow convergence.
- **Result**: Stuck at low accuracy (~12-15%) after many epochs.
- **Verdict**: **Too slow** for the number of classes....zzzzzz

### 2. Learning Rate = 0.05 (Aggressive)
- **Behavior**: Fast learning but significant overfitting.
- **Result**: 
  - Training Accuracy: ~89%
  - Validation Accuracy: ~62.5%
- **Verdict**: **Overfits.**

### 3. Learning Rate = 0.01 (Balanced)
- **Behavior**: Slower than 0.05 but more stable.
- **Result**: Reliable performance, but 0.05 achieved a higher peak validation accuracy before early stopping.
- **Verdict**: **The best.**

## Conclusion
Training on **58 granular classes** is difficult and hard to reach high accuracy like i hoped.
- **Learning Rate 0.05** remains the strongest performer for raw accuracy (~62.5%) in this configuration.
- **Learning Rate 0.01** is the best choice if we want reliability.
