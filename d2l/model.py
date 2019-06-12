"""The model module contains neural network building blocks"""
import torch

__all__ = ['corr2d', 'linreg']

def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def linreg(X, w, b):
	"""Linear regression."""
	return torch.mm(X,w) + b