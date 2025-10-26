# src/metrics.py
import torch

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def rse(pred, true):
    """
    Relative Squared Error (RSE):
    RSE = sqrt(sum((y_pred - y_true)^2)) / sqrt(sum((y_true - mean(y_true))^2))
    """
    numerator = torch.sqrt(torch.sum((pred - true) ** 2))
    denominator = torch.sqrt(torch.sum((true - torch.mean(true)) ** 2))
    return numerator / (denominator + 1e-8)

def corr(pred, true):
    """
    Pearson correlation between flattened predictions and ground truth.
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    pred_mean = torch.mean(pred_flat)
    true_mean = torch.mean(true_flat)

    num = torch.sum((pred_flat - pred_mean) * (true_flat - true_mean))
    denom = torch.sqrt(torch.sum((pred_flat - pred_mean) ** 2) * torch.sum((true_flat - true_mean) ** 2))
    return num / (denom + 1e-8)
