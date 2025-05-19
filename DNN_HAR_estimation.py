import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
epochs_har = 300

# Hyperparameters defined with hyperparameter_tuning.py
default_hp = {
    "hidden": (64, 64, 64),
    "lr": 0.001,
    "dropout": 0.15,
    "batch": 64,
    "epochs": 300,
}


########################################################################################
# Load data and define train, val, test split already in log format
########################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_prod.pkl", mode="rb"
) as f:
    df = pickle.load(f)


df_backup = df.copy()
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
    num_workers=0,
    prefetch_factor=1,
    persistent_workers=True,
):
    X_tr = torch.from_numpy(X[splits["train"]]).float()
    y_tr = torch.from_numpy(y[splits["train"]]).float()
    X_va = torch.from_numpy(X[splits["val"]]).float()
    y_va = torch.from_numpy(y[splits["val"]]).float()
    X_tr_va = torch.from_numpy(X[splits["train_val"]]).float()
    y_tr_va = torch.from_numpy(y[splits["train_val"]]).float()
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

    dl_tr_va = DataLoader(
        TensorDataset(X_tr_va, y_tr_va),
        batch_size=batch_size,
        shuffle=shuffle_train,
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
    return dl_tr, dl_va, dl_tr_va, dl_te


# Only 3 features
train_har_loader, val_har_loader, train_val_har_loader, test_har_loader = (
    create_data_loaders(X_har, y)
)
# Full 24 features (3 Features + 21 lags (RV_{t-2} ... RV_{t-22}))
train_full_loader, val_full_loader, train_val_full_loader, test_full_loader = (
    create_data_loaders(X_full, y)
)


########################################################################################
# Defining Loss function from models_tl.py with task_loss_optimized
########################################################################################

task_loss = task_loss_optimized(SR_vec, Gamma, k)
task_loss = task_loss.to(device)

########################################################################################
# Model training logic
########################################################################################


def fit(model, train_loader, val_loader, loss_fn, lr=0.001, epochs=100, verbose=True):
    model.to(device)
    # Selected via Hyperparameter tuning
    opt = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_loss_epoch_history = []
    val_loss_epoch_history = []

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
        train_loss_epoch_history.append(train_loss_epoch)

        model.eval()
        with torch.no_grad():
            val_losses = [
                loss_fn(model(xb.to(device)), yb.to(device)).item()
                for xb, yb in val_loader
            ]
        val_loss = float(np.mean(val_losses))
        val_loss_epoch_history.append(val_loss)

        # print every 10 epochs
        if verbose and ep % 10 == 0:
            print(
                f"epoch {ep}: train ={train_loss_epoch:.4e} val_loss = {val_loss:.4e}"
            )

        # Update best model according to validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Load best model in any epoch according to validation
    model.load_state_dict(best_state)

    history = {"train": train_loss_epoch_history, "val": val_loss_epoch_history}

    return model, history


########################################################################################
# HAR Models
########################################################################################


print("--------- HAR-MSE ---------")
har_mse, har_mse_hist = fit(
    StackedHAR(),
    train_har_loader,
    val_har_loader,
    mse_loss,
    lr=0.001,
    epochs=300,
    verbose=True,
)

for label, series in har_mse_hist.items():
    plt.plot(series[50:], label=label)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# Note the loss value is not directly interpretable since it is implemented as -U(\hat{v},y) and not U(v^*,y)-U(\hat{v},y therefore it is not centerd at 0
print("--------- HAR-TL ---------")
har_tl, har_tl_hist = fit(
    StackedHAR(),
    train_har_loader,
    val_har_loader,
    task_loss,
    lr=0.001,
    epochs=epochs_har,
    verbose=True,
)
for label, series in har_tl_hist.items():
    plt.plot(series[50:], label=label)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

########################################################################################
# DNN Models
########################################################################################


def build_and_fit(name, X_set, loaders, use_tl):
    p = hp_grid[name]
    neural_net = StackedDNN(
        in_features=X_set.shape[2],
        hidden_sizes=tuple(p["hidden"]),
        dropout=p["dropout"],
    )
    # Directly output the function
    return fit(
        neural_net,
        loaders[0],
        loaders[1],
        task_loss if use_tl else mse_loss,
        lr=p["lr"],
        epochs=p["epochs"],
    )


########################################################################################
# DNN Models for HAR dataset (d1)
########################################################################################

hp_grid = {
    name: default_hp for name in ["dnn_mse_d1", "dnn_tl_d1", "dnn_mse_d2", "dnn_tl_d2"]
}

print("--------- DNN-MSE-D1 ---------")
dnn_mse_d1, dnn_mse_d1_hist = build_and_fit(
    "dnn_mse_d1", X_har, (train_har_loader, val_har_loader), use_tl=False
)
for label, series in dnn_mse_d1_hist.items():
    plt.plot(series[50:], label=label)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

print("--------- DNN-TL-D1 ---------")
dnn_tl_d1, dnn_tl_d1_hist = build_and_fit(
    "dnn_tl_d1", X_har, (train_har_loader, val_har_loader), use_tl=True
)
for label, series in dnn_tl_d1_hist.items():
    plt.plot(series[50:], label=label)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

#######################################################################################
# DNN Models for Extended dataset (d2)
########################################################################################

print("--------- DNN-MSE-D2 ---------")
dnn_mse_d2, dnn_mse_d2_hist = build_and_fit(
    "dnn_mse_d2", X_full, (train_full_loader, val_full_loader), use_tl=False
)
for label, series in dnn_mse_d2_hist.items():
    plt.plot(series[50:], label=label)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

print("--------- DNN-TL-D2 ---------")
dnn_tl_d2, dnn_tl_d2_hist = build_and_fit(
    "dnn_tl_d2", X_full, (train_full_loader, val_full_loader), use_tl=True
)

for label, series in dnn_tl_d2_hist.items():
    plt.plot(series[50:], label=label)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

#######################################################################################
# Save & Load models
#######################################################################################

# model_list = {
#     "har_mse": har_mse,
#     "har_tl": har_tl,
#     "dnn_mse_d1": dnn_mse_d1,
#     "dnn_tl_d1": dnn_tl_d1,
#     "dnn_mse_d2": dnn_mse_d2,
#     "dnn_tl_d2": dnn_tl_d2,
# }

# # Save the full model
# for name, m in model_list.items():
#     torch.save(m, f"/Users/wanja/Developer/BA Thesis code/models/{name}.pth")

# Load models

# model_list = ["har_mse", "har_tl", "dnn_mse_d1", "dnn_tl_d1", "dnn_mse_d2", "dnn_tl_d2"]

# models_loaded = {}

# for name in model_list:
#     load_model = torch.load(
#         f"/Users/wanja/Developer/BA Thesis code/models/{name}.pth",
#         map_location=device,
#         weights_only=False,
#     )
#     models_loaded[name] = load_model

# har_mse, har_tl, dnn_mse_d1, dnn_tl_d1, dnn_mse_d2, dnn_tl_d2 = (
#     models_loaded[name]
#     for name in [
#         "har_mse",
#         "har_tl",
#         "dnn_mse_d1",
#         "dnn_tl_d1",
#         "dnn_mse_d2",
#         "dnn_tl_d2",
#     ]
# )

#######################################################################################
#######################################################################################
# Create forcasts for all 6 models and save in a df for analysis
########################################################################################
#######################################################################################


def forecast(model, test_loader):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            yb = model(xb).cpu()
            outs.append(yb)
    return torch.cat(outs).numpy()


y_test = y[splits["test"]]

preds = {
    "rv_true": y_test,
    "har_mse": forecast(har_mse, test_har_loader),
    "har_tl": forecast(har_tl, test_har_loader),
    "dnn_mse_d1": forecast(dnn_mse_d1, test_har_loader),
    "dnn_tl_d1": forecast(dnn_tl_d1, test_har_loader),
    "dnn_mse_d2": forecast(dnn_mse_d2, test_full_loader),
    "dnn_tl_d2": forecast(dnn_tl_d2, test_full_loader),
}

#######################################################################################
# Results Dataframe
#######################################################################################

dates = df["timestamp"].unique()[splits["test"]]

tickers = sorted(df["ticker"].unique())
# df for every model column (T,ticker)
dfs = {m: pd.DataFrame(a, index=dates, columns=tickers) for m, a in preds.items()}

# column values, ticker (swap to ticker, values)
df_results = pd.concat(dfs, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)

assert np.allclose(
    df_results["DIS", "rv_true"].values,
    df.loc[df.ticker == "DIS", "rv_target"][splits["test"]].values,
)

# Sanity
prob = []
for t in tickers:
    rv_true = df_results[t, "rv_true"].to_numpy()
    df_test = df_backup[df_backup["ticker"] == t]
    df_test = pd.DataFrame(df_test["rv_target"]).reset_index(drop=True)
    rv_target = df_test.loc[splits["test"], "rv_target"].to_numpy()
    if not np.allclose(rv_true, rv_target):
        prob.append(t)

print("Tickers with mismatches:", prob)


#######################################################################################
# Save results
#######################################################################################

# with open("/Users/Wanja/Developer/BA Thesis code/data/results.pkl", mode="wb") as f:
#     pickle.dump(df_results, f)

SR = SR_vec.cpu().numpy()
Gam = Gamma.cpu().numpy()
k = 2
#######################################################################################
#######################################################################################
# Plots and tables for thesis
########################################################################################
#######################################################################################
models = ["har_mse", "har_tl", "dnn_mse_d1", "dnn_tl_d1", "dnn_mse_d2", "dnn_tl_d2"]

log_true = df_results.xs("rv_true", level=1, axis=1)

per_stock = {}

for m in models:
    log_pred = df_results.xs(m, level=1, axis=1)

    # MSE
    # mse = ((np.exp(log_pred) - np.exp(log_true)) ** 2).mean(axis=0)
    mse = ((log_pred - log_true) ** 2).mean(axis=0)

    # Task Loss (exponentiates in the function)
    losses = task_loss_value(log_pred.values, log_true.values, SR, Gam, k)

    tl = pd.Series(losses.mean(axis=0), index=log_true.columns, name="TL")

    per_stock[m] = pd.DataFrame({"MSE": mse, "TL": tl})

tbl_stock = pd.concat(per_stock, axis=1)

# To latexx
print(
    tbl_stock.to_latex(
        caption=(
            "Stock level out-of-sample MSE and Taskloss for the 25 DJIA stocks across all models"
        ),
        label="tab:losses_models",
        float_format="%.3f",
        multicolumn=True,
        multicolumn_format="c",
        escape=False,
    )
)


#######################################################################################
# Cross summary
#######################################################################################

summary = {
    m: {"MSE_mean": per_stock[m]["MSE"].mean(), "TL_mean": per_stock[m]["TL"].mean()}
    for m in models
}

tbl_summary = pd.DataFrame(summary).T


print(
    tbl_summary.to_latex(
        label="tab:main_resutls1",
        caption="Cross‐sectional summary statistics for out-of-sample forecasts: MSE (log RV) and Taskloss (RV) of six models across 25 \gls{DJIA} stocks.",
        float_format="%.3f",
    )
)


#######################################################################################
# DM-Test with Newey west Hac t-stat
#######################################################################################

# Calcualte the Taskloss for each model for every day t in the test set
log_true_np = log_true.to_numpy(dtype=np.float64)

TL = {}
for m in ["har_mse", "har_tl", "dnn_mse_d1", "dnn_tl_d1", "dnn_mse_d2", "dnn_tl_d2"]:
    log_pred_np = preds[m].astype(np.float64)
    tl_1 = task_loss_value(preds[m], log_true_np, SR, Gam, k)
    TL[m] = pd.Series(tl_1, index=log_true.index, name=m)

# relevant pairs for rq
pairs = [
    ("har_mse", "har_tl"),
    ("dnn_mse_d1", "dnn_tl_d1"),
    ("dnn_mse_d2", "dnn_tl_d2"),
    ("har_tl", "dnn_tl_d1"),
]


results = []

# 10 lags like in patton and zhang
for m1, m2 in pairs:
    d = TL[m1] - TL[m2]
    X = np.ones((len(d), 1))
    model = sm.OLS(d.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

    meanΔ = model.params[0]
    t_HAC = model.tvalues[0]
    p_HAC = model.pvalues[0]

    results.append(dict(pair=f"{m1} vs {m2}", meanDiff=meanΔ, t=t_HAC, p=p_HAC))


results_table = pd.DataFrame(results).set_index("pair")

results_table2 = results_table.round(4)
print(
    results_table2.to_latex(
        float_format="%.3g",
        caption=(
            "Diebold–Mariano tests on daily task-loss differences with Newey–West standard errors"
        ),
        label="tab:main_results2",
    )
)


#######################################################################################
# RV_hat / RV_true ratio box plot
#######################################################################################
models = ["har_mse", "har_tl",
          "dnn_mse_d1", "dnn_tl_d1",
          "dnn_mse_d2", "dnn_tl_d2"]

titles = ["HAR-MSE", "HAR-TL",
          "DNN-MSE", "DNN-TL",
          "DNN-EXT-MSE", "DNN-EXT-TL"]

ratios = []
for m in models:
    log_hat  = df_results.xs(m,        level=1, axis=1).to_numpy(dtype=np.float64)
    log_true = df_results.xs("rv_true",level=1, axis=1).to_numpy(dtype=np.float64)
    ratios.append((np.exp(log_hat)/np.exp(log_true)).ravel())

plt.figure(figsize=(10, 4))
plt.boxplot(ratios,
            vert=True,               
            showfliers=True,
            widths=0.6)

plt.ylim(0, 15)                      
plt.xticks(ticks=np.arange(1, 7), labels=titles, rotation=45, ha="right")
plt.grid(axis='y', ls='--', alpha=.4)
plt.tight_layout()
plt.show()

# Only har 

models_two  = ["har_mse", "har_tl"]
titles_two  = ["HAR-MSE", "HAR-TL"]

ratios_two = []
for m in models_two:
    log_hat  = df_results.xs(m,        level=1, axis=1).to_numpy(dtype=np.float64)
    log_true = df_results.xs("rv_true",level=1, axis=1).to_numpy(dtype=np.float64)
    ratios_two.append((np.exp(log_hat) / np.exp(log_true)).ravel())


plt.figure(figsize=(5, 4))
plt.boxplot(ratios_two,
            vert=True,         
            showfliers=True,
            widths=0.6)

plt.ylim(0, 15)
plt.xticks(ticks=[1, 2], labels=titles_two)
plt.grid(axis='y', ls='--', alpha=.4)
plt.tight_layout()
plt.show()

#######################################################################################
#######################################################################################
# Quick plot
########################################################################################
#######################################################################################
from result_analysis import plot_df_results, plot_df_results_single

df_results_exp = np.exp(df_results)


plot_df_results_leg(df=df_results, all=False)

plot_df_results_leg(df=df_results_exp, all=False)

plot_df_results_single(df=df_results_exp, ticker="CRM", all=False)


#######################################################################################
#######################################################################################
# Robustnesss analysis HAR DNN
########################################################################################
#######################################################################################


tickers = sorted(df["ticker"].unique())
ols_params = np.zeros((len(tickers), 4))

# Estimate traditional OLS HAR models

for j, tic in enumerate(tickers):
    # train + val, J stock, 3 inputs
    Xj = X_har[splits["train_val"], j, :]
    yj = y[splits["train_val"], j]
    Xj = sm.add_constant(Xj)
    ols_params[j, :] = sm.OLS(yj, Xj).fit().params

ols_df = pd.DataFrame(
    ols_params, index=tickers, columns=["const", "rv_d", "rv_w", "rv_m"]
)

# Get coefficients from torch models
weights = np.vstack([m.weight.detach().cpu().numpy() for m in har_mse.hars])
biases = np.array([m.bias.detach().cpu().item() for m in har_mse.hars])

har_df = pd.DataFrame(weights, index=tickers, columns=["rv_d", "rv_w", "rv_m"])
har_df["const"] = biases
har_df = har_df[["const", "rv_d", "rv_w", "rv_m"]]


# Save both parameters
with open(
    "/Users/wanja/Developer/BA Thesis code/data/har_param_comparison.pkl", "wb"
) as f:
    pickle.dump({"har_mse": har_df, "har_ols": ols_df}, f)


cols = ["const", "rv_d", "rv_w", "rv_m"]
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=120)

for ax, col in zip(axes.ravel(), cols):
    ax.scatter(har_df[col], ols_df[col], s=30, color="#0000FF")
    lim = np.array([har_df[col].min(), har_df[col].max()])
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(f"HAR_MSE {col}")
    ax.set_ylabel(f"OLS {col}")
    ax.set_title(col)

plt.tight_layout()
# plt.savefig("/Users/wanja/Developer/BA Thesis code/figures/param_compare_all.png")
plt.show()

#######################################################################################
#######################################################################################
# Robustnesss analysis for k
########################################################################################
#######################################################################################


k_grid = [0.5, 1, 2, 3]
series = {}
y_test = y[splits["test"]]

for k_val in k_grid:
    tl_loss = task_loss_optimized(SR_vec, Gamma, k_val).to(device)

    for name, build in {
        "har_tl": StackedHAR,
        "dnn_tl_d1": lambda: StackedDNN(
            in_features=X_har.shape[2], hidden_sizes=(64, 32)
        ),
        "dnn_tl_d2": lambda: StackedDNN(
            in_features=X_full.shape[2], hidden_sizes=(64, 32)
        ),
    }.items():
        is_d1 = ("d1" in name) or (name == "har_tl")
        train_l = train_har_loader if is_d1 else train_full_loader
        val_l = val_har_loader if is_d1 else val_full_loader
        X_test = X_har[splits["test"]] if is_d1 else X_full[splits["test"]]

        model = fit(
            build(), train_l, val_l, tl_loss, lr=0.001, epochs=75, verbose=False
        )

        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().to(device)
            y_hat = model(X_test_t).cpu().numpy()

        util = task_loss_value(
            y_hat,
            y_test,
            SR_vec.cpu().numpy(),
            Gamma.cpu().numpy(),
            k_val,
        )
        series[(name, k_val)] = util.mean()

pd.Series(series).unstack(0).plot(marker="o")
plt.ylabel("Taskloss (lower is better)")
plt.title("TL for different risk-aversion paramters")
plt.savefig("figures/k_robustness.svg", format="svg")
plt.savefig("figures/k_robustness.png", format="png")
plt.show()
