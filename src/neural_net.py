import numpy as np
import torch 
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class TorchMLRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_size=128,
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-3,
        epochs=200,
        random_state=42,
    ):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        self.y_mean_ = y_t.mean()

        self.net_ = nn.Sequential(
            nn.Linear(X_t.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            self.net_.train()
            optimizer.zero_grad()
            loss = loss_fn(self.net_(X_t), y_t - self.y_mean_)
            loss.backward()
            optimizer.step()
            
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            predictions = self.net_(X_t) + self.y_mean_
        return predictions.numpy().ravel()