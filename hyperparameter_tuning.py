import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import time
import matplotlib.pyplot as plt

# Import from project file
from model_tl import (
    task_loss_value,
    StackedHAR,
    StackedDNN,
    task_loss_optimized,
    mse_loss,
)

device = torch.device("mps")


########################################################################################
# Helper function to correctly arrange orignal df
########################################################################################


def create_array(df, feat_cols):
    # MultiIndex columns
    wide = df.pivot(index="timestamp", columns="ticker", values=feat_cols)
    # swap columns multiindex column to (ticker, feature)
    wide = wide.swaplevel(0, 1, axis=1)

    tickers = sorted(wide.columns.levels[0])
    features = feat_cols

    desired_cols = pd.MultiIndex.from_product(
        [tickers, features], names=["ticker", "feature"]
    )

    # Reoder to the correct setting  aapl rv_d, aapl, rv_w, aapl, rv_m , amgn rv_d, amgn rv_w ...
    wide = wide.reindex(columns=desired_cols)

    # reshape T and Ftot (3*25 for X, 25 for y)
    T, Ftot = wide.shape
    nt = wide.columns.levels[0].size
    nf = len(feat_cols)
    assert Ftot == nt * nf, f"Expected {nt * nf} columns, got {Ftot}"
    return wide.values.reshape(T, nt, nf)


########################################################################################
# Definition of constant SR and Riskaversion parameter k
########################################################################################

k = 2
epochs_har = 200


########################################################################################
# Load data and define train, val, test split
########################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_prod.pkl", mode="rb"
) as f:
    df = pickle.load(f)


########################################################################################


nlags = 22
df = df[
    [
        "timestamp",
        "ticker",
        "rv_target",
        "rv_d",
        "rv_w",
        "rv_m",
        *[f"lag_{i}" for i in range(2, (nlags + 1))],
    ]
]

# Shape is (timestamp, number of stocks, number of featurs)
X_har = create_array(df, ["rv_d", "rv_w", "rv_m"])

X_full = create_array(df, ["rv_d", "rv_w", "rv_m", *[f"lag_{i}" for i in range(2, 23)]])

y = (
    df.pivot(index="timestamp", columns="ticker", values="rv_target")
    .sort_index(axis=1)
    .values
)

# y= (
#     df.pivot(index="timestamp", columns="ticker", values="rv_target")
#     .values
# )


# Train 70%, Val 15%, Test 15%
T = X_har.shape[0]
T_train = int(T * 0.7)
T_val = int(T * 0.85)

splits = {
    "train": slice(0, T_train),
    "val": slice(T_train, T_val),
    "train_val": slice(0, T_val),
    "test": slice(T_val, T),
    "all": slice(0, T),
}


########################################################################################
# Estimating correlation for train and val set for training and on full for test
########################################################################################


# Load Gamma and SR
with open("/Users/wanja/Developer/BA Thesis code/data/SR_vec.pkl", "rb") as f:
    SR_vec = pickle.load(f)

SR_vec = torch.from_numpy(SR_vec.values).float()

with open("/Users/wanja/Developer/BA Thesis code/data/Gamma.pkl", "rb") as f:
    Gamma = pickle.load(f)

Gamma = torch.from_numpy(Gamma).float()

########################################################################################
# Data Loader
########################################################################################


def create_data_loaders(
    X,
    y,
    batch_size=64,
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False,
    num_workers=6,
    prefetch_factor=1,
    persistent_workers=True,
):
    X_tr = torch.from_numpy(X[splits["train"]]).float()
    y_tr = torch.from_numpy(y[splits["train"]]).float()
    X_va = torch.from_numpy(X[splits["val"]]).float()
    y_va = torch.from_numpy(y[splits["val"]]).float()
    X_te = torch.from_numpy(X[splits["test"]]).float()
    y_te = torch.from_numpy(y[splits["test"]]).float()

    dl_tr = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=bool(num_workers),
    )
    dl_va = DataLoader(
        TensorDataset(X_va, y_va),
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=bool(num_workers),
    )
    dl_te = DataLoader(
        TensorDataset(X_te, y_te),
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=bool(num_workers),
    )
    return dl_tr, dl_va, dl_te


# Only 3 features
train_har_loader, val_har_loader, test_har_loader = create_data_loaders(X_har, y)
# Full 24 features (3 Features + 21 lags (RV_{t-2} ... RV_{t-22}))
train_full_loader, val_full_loader, test_full_loader = create_data_loaders(X_full, y)


########################################################################################
# Defining Loss function from models_tl.py with task_loss_optimized
########################################################################################

task_loss = task_loss_optimized(SR_vec, Gamma, k)
task_loss = task_loss.to(device)

########################################################################################
# Model training logic
########################################################################################


def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    lr=0.001,
    epochs=100,
    verbose=True,
    patience=15,
):
    model.to(device)

    # model = torch.compile(model)
    # Selected via Hyperparameter tuning
    opt = optim.Adam(model.parameters(), lr=lr)
    best, best_state = float("inf"), None

    no_improve = 0
    for ep in range(1, epochs + 1):
        # training
        model.train()
        train_loss = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # flush previous gradients
            opt.zero_grad()
            # calc loss
            loss = loss_fn(model(xb), yb)
            # calc derivative w.r.t weights
            loss.backward()
            # update weights with lr
            opt.step()
            train_loss.append(loss.item())

        train_loss_epoch = float(np.mean(train_loss))

        # validation
        with torch.no_grad():
            model.eval()
            val_loss = np.mean(
                [
                    loss_fn(model(xb.to(device)), yb.to(device)).item()
                    for xb, yb in val_loader
                ]
            )
        # Update every 10 epoches
        if verbose and ep % 10 == 0:
            print(
                f"epoch {ep}: train ={train_loss_epoch:.4e} val_loss = {val_loss:.4e}"
            )

        # Update best model according to validation loss and early stopping for grid search
        if val_loss < best:
            best, best_state = (
                val_loss,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"→ stopping early at epoch {ep} (no improvement in {patience} epochs)"
            )
            break
    # Load best model in any epoch according to validation
    model.load_state_dict(best_state)
    return model


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# Hyperparameter part
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

import itertools
import json
from copy import deepcopy
from tqdm.auto import tqdm

#######################################################################################
# Define HP grid
# Here with early stopping!
# Only for D2 and TL due to computational costs
#######################################################################################

# Test later with the best model
#     "optim": ["adam", "sgd", "rmsprop"],

GRID = {
    "hidden": [
        (128, 64),
        (64, 64, 64),
        (128, 64, 32),
        (128, 128, 64, 64),
        (256, 128, 128, 128),
        (256, 256, 128, 128, 128),
    ],
    "lr": [0.0001, 0.001, 0.01],
    "dropout": [0, 0.15, 0.4],
    "batch": [32, 64],
    "epochs": [150, 300],
}


def cartesian(d):
    keys = d.keys()
    for vals in itertools.product(*d.values()):
        yield dict(zip(keys, vals))


#######################################################################################
# data from main
#######################################################################################


# So loader is hp
def loaders(batch):
    return create_data_loaders(X_full, y, batch_size=batch, shuffle_train=True)


#######################################################################################
# Evalue
#######################################################################################

loss_fn = task_loss_optimized(SR_vec, Gamma, k).to(device)


def evaluate(p):
    # NO test set for hypeparameter tuning eval at the end with val
    tr, va, _ = loaders(p["batch"])
    net = StackedDNN(
        in_features=X_full.shape[2], hidden_sizes=p["hidden"], dropout=p["dropout"]
    ).to(device)

    # Same as in main
    net = fit(net, tr, va, loss_fn, lr=p["lr"], epochs=p["epochs"], verbose=True)

    # Compute the val loss again for the best model and save this
    net.eval()
    with torch.no_grad():
        val_loss = np.mean(
            [loss_fn(net(xb.to(device)), yb.to(device)).item() for xb, yb in va]
        )
    return val_loss


OUTFILE = "/Users/wanja/Developer/BA Thesis code/dnn_best_tmp.json"

best_loss, best_p = 1e9, None
params = list(cartesian(GRID))

for p in tqdm(params, desc="Grid-Search", leave=False):
    try:
        loss = evaluate(p)
    except Exception as e:
        print("Evaluation failed for", p, ":", e)
        continue

    if loss < best_loss:
        best_loss, best_p = loss, deepcopy(p)
        print("new best:", best_loss, best_p)

        # -
        with open(OUTFILE, "w") as f:
            json.dump({"val_loss": best_loss, "hp": best_p}, f, indent=2)


print("Selected Hyper-parameters:", best_p)
json.dump(
    best_p, open("/Users/wanja/Developer/BA Thesis code/dnn_best.json", "w"), indent=2
)
