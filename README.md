# Neural Network Function Approximation Experiments

## Overview
This repository contains PyTorch implementations for neural network-based function approximation experiments. Each folder corresponds to a specific experiment setup as presented in the accompanying research paper.

## Environment Requirements
- **Python 3.8+**
- **Core Dependencies:**
  - Torch: 2.6.0+cu124
  - Scipy: 1.15.2
  - Numpy: 2.2.3

## Project Structure

experiments/
- linearApprox: Approximation to three-dimensional linear functions;
- pieceApprox: Approximation to piece-wise functions;
-  sinApprox: Approximation to sin(2x) and sin(2x) + sin(6x) + sin(10x)  functions;
- linear_pretrain: Pre-training on one-dimensional linear function approximation
- sinx_interpolation: Sine interpolation experiments;
- func_img: Image processing experiments;
- func_video: Video processing experiments.




## Experiment Folder Structure
Each experiment folder contains:
- `model_x1/`, `model_x2/` - Trained model parameters
- `results/` - Experimental results and loss values
- `main.py` - Main execution script
- `data_generation.py` - Training data generation
- `NNnetwork.py` - Neural network architecture
- `train_class.py` - Training procedure implementation
- `save_data.py` - Data persistence utilities
- `save_test_loss.py` - Loss recording functionality

## Quick Start
1. Install dependencies:
```bash
pip install torch==2.6.0 scipy==1.15.2 numpy==2.2.3
```
