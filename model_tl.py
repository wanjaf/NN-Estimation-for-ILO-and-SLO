import torch
import torch.nn as nn
import pandas as pd
import numpy as np

#####################################################################################################
# Model Defintions
#####################################################################################################

# HAR model is a stacked NN with 25 single layer NN with three neurons and bias
# Replicating 28 distinct and seperated HAR models


class StackedHAR(nn.Module):
    # 25 distinct HAR models for the 25 stocks
    def __init__(self, n_series=25):
        super().__init__()
        # one Linear per stock rv
        self.hars = nn.ModuleList([nn.Linear(3, 1, bias=True) for _ in range(n_series)])

    def forward(self, x):
        # Input is (t,25,3) and output is (t,25)
        outs = []
        for i, har in enumerate(self.hars):
            # forward pass for all t, stock i and rv_d, rv_w, rv_m
            outs.append(har(x[:, i, :]))
        return torch.cat(outs, dim=1)


# Stacked DNN
# Isolated DNN for each individual stock with layer num and neuron num as hyperparemter + more


class StackedDNN(nn.Module):
    def __init__(
        self, in_features, hidden_sizes, n_series=25, activation=nn.ReLU, dropout=0.0
    ):
        super().__init__()
        # 25 sequential models
        self.models = nn.ModuleList()

        for _ in range(n_series):
            layers = []
            prev_dim = in_features

            # Hidden Layers
            for h in hidden_sizes:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = h

            # Final output layer for 1 prediction for each model in n_series
            layers.append(nn.Linear(prev_dim, 1))

            # Package each DNN in a single module (stacked DNN)
            self.models.append(nn.Sequential(*layers))

    def forward(self, x):
        # Input is (t,25,n_features) and output is (t,25)
        outs = []

        # Loop over all dnn in stack
        for i, dnn in enumerate(self.models):
            # All data for the batch for the I'th stock and all features
            x_i = x[:, i, :]
            # Pass through dnn and create prediction
            y_i = dnn(x_i)
            outs.append(y_i)

        # Consolidate all predictions for input in loss function
        return torch.cat(outs, dim=1)


#####################################################################################################
#  Loss functions
#####################################################################################################

#####################################################################################################
# Taskloss numpy implementation
#####################################################################################################


def task_loss_value(log_sigma_hat, log_sigma_true, SR, Gamma, k):
    sigma_hat = np.exp(log_sigma_hat)
    sigma = np.exp(log_sigma_true)
    Gamma_inv = np.linalg.inv(Gamma)
    # Optimal utility
    util_opt = (SR @ (Gamma_inv @ SR)).sum() / (2 * k)

    losses = []

    for y_hat, y_true in zip(sigma_hat, sigma):
        r = y_true / y_hat
        D = np.diag(r)

        # Realized utiliyt
        term1 = SR @ (Gamma_inv @ (D @ SR)) / k
        term2 = SR @ (Gamma_inv @ (D @ (Gamma @ (D @ (Gamma_inv @ SR))))) / (2 * k)

        util_realized = term1 - term2
        loss = util_opt - util_realized

        losses.append(loss)
    return np.array(losses)


def realized_utility_compute(log_sigma_hat, log_sigma_true, SR, Gamma, k):
    sigma_hat = np.exp(log_sigma_hat)
    sigma = np.exp(log_sigma_true)
    Gamma_inv = np.linalg.inv(Gamma)

    utils = []

    for y_hat, y_true in zip(sigma_hat, sigma):
        r = y_true / y_hat
        D = np.diag(r)

        # Realized utiliyt
        term1 = SR @ (Gamma_inv @ (D @ SR)) / k
        term2 = SR @ (Gamma_inv @ (D @ (Gamma @ (D @ (Gamma_inv @ SR))))) / (2 * k)

        util_realized = term1 - term2

        utils.append(util_realized)
    return np.array(utils)

#####################################################################################################
# Taskloss torch implementation
#####################################################################################################

mse_loss = nn.MSELoss()

# See 2.27 in the thesis. However since the constant optimal utility is constant one can just optimize
# Eq. 2.26 namely - U(\mathbf{\hat{v}}_t; \sigma_{t+1})
# W_f, 1 and R_f are dropped since they they get dropped during back propagation


class task_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_sigma_hat, sigma, SR, Gamma, k):
        # exponentiate the log (sigma_hat and sigma are of shape (t,25))
        sigma_hat = torch.exp(log_sigma_hat)
        sigma = torch.exp(sigma)
        batch, n = sigma.shape

        # All on device
        dev = sigma.device
        SR = SR.to(dev)
        Gamma = Gamma.to(dev)

        # inverse of Gamma, can technically also be done outside of the loss since constant
        # But doing it inside allows for later implementation with different correaltion (maybe rolling?)
        # Optinal if to slow: Pass Gamma inverse form outside since we assume it as constant - correlation targetting
        Gamma_inv = torch.linalg.inv(Gamma)

        utilities = []
        for t in range(batch):
            # # Compute diag(sigma_true/sigma_pre)
            r = sigma[t] / sigma_hat[t]
            D = torch.diag(r)

            first = SR @ (Gamma_inv @ (D @ SR)) / k
            second = SR @ (Gamma_inv @ (D @ (Gamma @ (D @ (Gamma_inv @ SR))))) / (2 * k)

            realized_util = first - second
            utilities.append(realized_util)

        utils = torch.stack(utilities)
        return -utils.mean()


class task_loss_optimized_old(nn.Module):
    def __init__(self, SR, Gamma, k):
        super().__init__()
        self.register_buffer("SR", SR)
        self.register_buffer("Gamma", Gamma)
        self.register_buffer("Gamma_inv", torch.linalg.inv(Gamma))
        self.register_buffer("Gamma_inv_SR", self.Gamma_inv @ SR)
        self.k = k

    def forward(self, log_sigma_hat, sigma):
        sigma_hat = torch.exp(log_sigma_hat)  # (B,n)
        sigma = torch.exp(sigma)
        w = sigma / sigma_hat  # (B,n)
        term1 = (self.SR * (w * self.Gamma_inv_SR)).sum(1) / self.k
        temp = w * self.Gamma_inv_SR  # (B,n)
        gtemp = temp @ self.Gamma  # (B,n)
        term2 = (temp * gtemp).sum(1) / (2 * self.k)
        U = term1 - term2  # (B,)
        return -U.mean()


class task_loss_optimized(nn.Module):
    def __init__(self, SR, Gamma, k):
        super().__init__()
        self.register_buffer("SR", SR)
        self.register_buffer("Gamma", Gamma)
        self.register_buffer("Gamma_inv", torch.linalg.inv(Gamma))
        self.register_buffer("Gamma_inv_SR", self.Gamma_inv @ SR)
        self.k = k

        # Calculate optimal utility so loss is easier interpretable (goes to 0 as model gets better)
        opt = (SR * self.Gamma_inv_SR).sum() / (2 * k)
        self.register_buffer("U_opt", opt.unsqueeze(0))

    def forward(self, log_sigma_hat, sigma):
        sigma_hat = torch.exp(log_sigma_hat)  # (B,n)
        sigma = torch.exp(sigma)
        w = sigma / sigma_hat  # (B,n)
        term1 = (self.SR * (w * self.Gamma_inv_SR)).sum(1) / self.k
        temp = w * self.Gamma_inv_SR  # (B,n)
        gtemp = temp @ self.Gamma  # (B,n)
        term2 = (temp * gtemp).sum(1) / (2 * self.k)
        U = term1 - term2  # (B,)
        loss = (self.U_opt - U).mean()
        return loss


#####################################################################################################
# Test of loss function
#####################################################################################################

# # 1 Asset

# import pickle

# with open(
#     "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
# ) as f:
#     df = pickle.load(f)

# df = np.log(df)

# log_sigma_hat = df.iloc[0, 0]
# sigma = df.iloc[1, 0]

# log_sigma_hat = np.log(0.005)
# sigma = np.log(0.015)


# SR = 0.4
# k = 0.5
# Gamma = 1

# sigma_hat = np.exp(log_sigma_hat)
# sigma_exp = np.exp(sigma)

# # 1 Asset loss
# ratio = sigma_exp / sigma_hat
# term1 = (SR**2 / k) * ratio
# term2 = (SR**2 / (2 * k)) * ratio**2

# loss_np = -(term1 - term2)

# log_sigma_hat_t = torch.tensor(
#     [[log_sigma_hat]], dtype=torch.float32, requires_grad=True
# )
# sigma_t = torch.tensor([[sigma]], dtype=torch.float32)
# SR_t = torch.tensor([SR], dtype=torch.float32)
# Gamma_t = torch.tensor([[Gamma]], dtype=torch.float32)

# log_sigma_hat_t2 = log_sigma_hat_t

# loss_fn = task_loss()
# loss_pt = loss_fn(log_sigma_hat_t, sigma_t, SR_t, Gamma_t, k)
# loss_pt.backward()
# grad_val = log_sigma_hat_t.grad.item()

# log_sigma_hat_t.grad.zero_()
# loss_fn_op = task_loss_optimized(SR_t, Gamma_t, k)
# loss_pt_op = loss_fn_op(log_sigma_hat_t2, sigma_t)
# loss_pt_op.backward()
# grad_val_opt = log_sigma_hat_t2.grad.item()

# print(f"NumPy loss:            {loss_np:.6f}")
# print(f"PyTorch loss tensor:   {loss_pt.item():.6f}")
# print(f"Gradient dL/d(log sigma)):   {grad_val:.6f}")
# print(f"PyTorch optimized loss tensor:   {loss_pt_op.item():.6f}")
# print(f"Gradient optimized dL/d(log sigma)):   {grad_val_opt:.6f}")


# Multi Asset Test

# # Test parameters
# B, n = 64, 25
# log_sigma_true = torch.randn(B, n)
# sigma_true = torch.exp(log_sigma_true)
# log_sigma_base = torch.randn(B, n)
# SR = torch.randn(n)
# SR_vec = SR.unsqueeze(0).repeat(B, 1)
# Gamma_raw = torch.randn(n, n)
# Gamma = (Gamma_raw @ Gamma_raw.T) / n + 0.05 * torch.eye(n)
# k = 0.5

# loss_numpy = task_loss_value(
#     log_sigma_base.numpy(), log_sigma_true.numpy(), SR.numpy(), Gamma.numpy(), k
# )

# loss_numpy_mean = loss_numpy.mean()

# log_sigma_hat2 = log_sigma_base.clone().detach().requires_grad_(True)
# loss_opt = task_loss_optimized(SR, Gamma, k)(log_sigma_hat2, log_sigma_true)

# loss_numpy_t = torch.tensor(loss_numpy_mean, dtype=loss_opt.dtype, device=loss_opt.device)
# torch.testing.assert_close(loss_opt, loss_numpy_t, rtol=1e-5, atol=1e-7)
# print(f'Match: {None == torch.testing.assert_close(loss_opt, loss_numpy_t, rtol=1e-5, atol=1e-7)}')
