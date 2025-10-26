# Lympha (Temporal GCN-RL for Water Quality Forecasting)

An end-to-end PyTorch implementation of the Temporal Graph Convolutional Autoencoder with Reinforcement Learning adjacency refinement (Lympha), adapted for multi-parameter water quality forecasting.

## Features
- Temporal encoder via 1D Conv
- Multi-graph (static, attention, RL) fusion
- Variational Autoencoder (STAE)
- RL-based dynamic graph refinement
- Evaluation metrics (MAE, RMSE, RSE, CORR)
- Synthetic and real dataset ready

## Quickstart
```bash
python -m scripts.prepare_data
python -m scripts.train_min
python -m scripts.train_stae
python -m scripts.train_rl
python -m scripts.eval_and_plot
