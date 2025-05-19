import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams["svg.fonttype"] = "none"

# Color scheme
# Blue = #0000FF
# Red = #FF0000
# Green = #008002
# Green2 = #00FF00

###############################################################################
# Function definition


def f_utility(SR, k, y_true, y_pred):
    utility = ((np.square(SR) / k) * (y_true / y_pred)) - (
        (np.square(SR) / (2 * k)) * (np.square(y_true) / np.square(y_pred))
    )
    return 1 + utility


def f_custom_loss(SR, k, y_true, y_pred):
    risk_target_reward = np.square(SR) / k
    variance_scaling = np.square(y_pred - y_true) / (np.square(y_pred) * 2)
    utility = risk_target_reward * variance_scaling
    return utility


def f_mse(y_true, y_pred):
    return np.square(y_true - y_pred)


SR, k = 0.4, 2

# Create  meshgrid of realized volatility values (for true and "predicted")
min_range, max_range, step_size = 0.01, 0.03, 0.0005


y_true = np.arange(min_range, max_range, step_size)
y_pred = np.arange(min_range, max_range, step_size)
y_true, y_pred = np.meshgrid(y_true, y_pred)

z_util = f_utility(SR=SR, k=k, y_true=y_true, y_pred=y_pred)
z_loss = f_custom_loss(SR=SR, k=k, y_true=y_true, y_pred=y_pred)
z_mse = f_mse(y_true=y_true, y_pred=y_pred)


####################################################################################
# Visualization of the Utility function
#################################################

# Set up figure and axis
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# using coolwarm
surf = ax.plot_surface(y_true, y_pred, z_util, cmap="coolwarm")

# rotate labels
ax.xaxis.set_tick_params(rotation=45)  # Rotate x-axis labels
ax.yaxis.set_tick_params(rotation=-45)  # Rotate y-axis labels

# Adding labels
# ax.set_title('Visualization of the Utility Function')
ax.set_xlabel("True Volatility", labelpad=20)
ax.set_ylabel("Predicted Volatility", labelpad=20)
ax.set_zlabel("Realized Utility", labelpad=15)

ax.set_xlim(min_range, max_range)
ax.set_ylim(max_range, min_range)

# Add color bar for reference
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
plt.savefig("figures/utility_function.svg", format="svg")
plt.show()


###################################################################################
# Visualization of MSE
#####################################################################################################

# Plot MSE

# Set up figure and axis
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# using coolwarm
z_mse = z_mse * 1e3


surf = ax.plot_surface(y_true, y_pred, z_mse, cmap="coolwarm")

# rotate labels
ax.xaxis.set_tick_params(rotation=45)  # Rotate x-axis labels
ax.yaxis.set_tick_params(rotation=-45)  # Rotate y-axis labels

# Adding labels
# ax.set_title('Visualization of the Error Weighting by MSE')
ax.set_xlabel("True Volatility", labelpad=20)
ax.set_ylabel("Predicted Volatility", labelpad=20)
ax.set_zlabel("Loss", labelpad=15)

ax.set_xlim(min_range, max_range)
ax.set_ylim(max_range, min_range)

# Add color bar for reference
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15)
plt.savefig("figures/mse_loss_function.svg", format="svg")
plt.show()


##############################################################################
# Plot the Loss function
#################################################################################
# Set up figure and axis
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# using coolwarm
surf = ax.plot_surface(y_true, y_pred, z_loss, cmap="coolwarm")

# rotate labels
ax.xaxis.set_tick_params(rotation=45)
ax.yaxis.set_tick_params(rotation=-45)

# Adding labels
# ax.set_title('Visualization of Error Weighting by the Custom Loss Function')
ax.set_xlabel("True Volatility", labelpad=20)
ax.set_ylabel("Predicted Volatility", labelpad=20)
ax.set_zlabel("Loss", labelpad=15)

ax.set_xlim(min_range, max_range)
ax.set_ylim(max_range, min_range)

# Add color bar for reference
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15)
plt.savefig("figures/custom_loss_function.svg", format="svg")
plt.show()


##############################################################################
# Visualization of distribution normal + log for a complete aggregate series (all 25 combined)
##############################################################################
# Not used in thesis

# Color scheme
# Blue = #0000FF
# Red = #FF0000
# Green = #008002
# Green2 = #00FF00

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

# Distribution visualization by aggergating all stocks
# all in one array for aggregation
all_rv = df.values.flatten()
# Plausibility check
0 == sum(all_rv <= 0)

# Log transformation
all_log_rv = np.log(df).values.flatten()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# raw RV histogram
axes[0].hist(all_rv, bins=100, density=True, color="#0000FF")
axes[0].set_xlabel("Realized Volatility")

# log-RV histogram
axes[1].hist(all_log_rv, bins=100, density=True, color="#0000FF")
axes[1].set_xlabel("Log Realized Volatility")

plt.tight_layout()

# save as SVG
plt.savefig("figures/descriptive_part/histogram.svg", format="svg")
plt.show()


#################################################################################
# ACF Plots log-rv
#################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

df = np.log(df)

df.columns = df.columns.droplevel(0)
tickers = df.columns

nlags = 50

fig, axes = plt.subplots(7, 4, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()

# loop through each ticker
for ax, ticker in zip(axes, tickers):
    series = df[ticker]
    plot_acf(
        df[ticker],
        lags=nlags,
        ax=ax,
        zero=False,
        bartlett_confint=False,
        color="#0000FF",
        vlines_kwargs={"color": "#0000FF"},
    )
    ax.set_title(f"ACF of {ticker}")
    ax.set_ylim(-0.3, None)


plt.tight_layout()
plt.savefig("figures/descriptive_part/acf_plots.svg", format="svg")
plt.savefig("figures/descriptive_part/acf_plots.png", format="png")
plt.show()


########################################################################################
# Random acf for main text
########################################################################################

tickers_short = tickers[[6, 8, 13, 20]]

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.flatten()


# loop through each ticker
for ax, ticker in zip(axes, tickers_short):
    series = df[ticker]
    plot_acf(
        df[ticker],
        lags=nlags,
        ax=ax,
        zero=False,
        bartlett_confint=False,
        color="#0000FF",
        vlines_kwargs={"color": "#0000FF"},
    )
    ax.set_title(f"ACF of {ticker}")
    ax.set_ylim(-0.2, None)


plt.tight_layout()
plt.savefig("figures/descriptive_part/acf_plot_main.svg", format="svg")
plt.show()


#########################################################################################
# Distribution: normal vs log for 4 individual stocks
#########################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

df.columns = df.columns.droplevel(0)

# Blue = #0000FF
# Red = #FF0000
# Green = #008002
# Orange = #FE9928

tickers = ["CSCO", "DIS", "INTC", "MSFT"]
colors = ["#FE9928", "#FF0000", "#008002", "#0000FF"]
# get rv & log  rv series
data = {ticker: df[ticker] for ticker in tickers}
logdata = {ticker: np.log(data[ticker]) for ticker in tickers}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# Left panel: RV histograms
for ticker, col in zip(tickers, colors):
    axes[0].hist(
        data[ticker], bins=50, density=True, alpha=0.7, color=col, label=ticker
    )
axes[0].set_title("Realized Volatility")
axes[0].set_xlabel("RV")
axes[0].legend()

# Right panel: log-RV histograms
for ticker, col in zip(tickers, colors):
    axes[1].hist(
        logdata[ticker], bins=40, density=True, alpha=0.7, color=col, label=ticker
    )
axes[1].set_title("Log Realized Volatility")
axes[1].set_xlabel("log-RV")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/descriptive_part/density_4_stocks.svg", format="svg")
plt.show()


#################################################################################
# Descriptive statistics for all 25 RV with log transformation
#################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

# Convert to natural logarithm of RV
df_log_rv = np.log(df)

# Using scipy to create a summary table for all 25 log RV

descriptive_for_stock_log_rv = pd.DataFrame(
    {
        "Mean": df_log_rv.mean(),
        "Std.": df_log_rv.std(),
        # No bias correction is implemented just standard skewness
        "Skew.": df_log_rv.skew(),
        # NO bias correction is implemented just standard kurtosis
        "Ex.Kurt": df_log_rv.kurtosis(),
        "Median": df_log_rv.median(),
        "Min": df_log_rv.min(),
        "0.25": df_log_rv.quantile(0.25),
        "0.75": df_log_rv.quantile(0.75),
        "Max": df_log_rv.max(),
        "JB p-value": jarque_bera(x=df_log_rv, axis=0, keepdims=False)[1],
    }
)

descriptive_for_stock_log_rv = descriptive_for_stock_log_rv.droplevel(0)


# Output for latex
print(
    descriptive_for_stock_log_rv.to_latex(
        label="tab:descriptive_statistics_log_rv",
        caption="Stock level summary statistics for log realized volatilities",
        float_format="%.3f",
    )
)


#################################################################################
# Descriptive statistics for all 25 RV without log transformation
#################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

# Using scipy to create a summary table for all 25 log RV

descriptive_for_stock_rv = pd.DataFrame(
    {
        "Mean": df.mean(),
        "Std.": df.std(),
        # No bias correction is implemented just standard skewness
        "Skew.": df.skew(),
        # NO bias correction is implemented just standard kurtosis
        "Ex.Kurt": df.kurtosis(),
        "Median": df.median(),
        "Min": df.min(),
        "0.25": df.quantile(0.25),
        "0.75": df.quantile(0.75),
        "Max": df.max(),
        "JB p-value": jarque_bera(x=df, axis=0, keepdims=False)[1],
    }
)

descriptive_for_stock_rv = descriptive_for_stock_rv.droplevel(0)


print(
    descriptive_for_stock_rv.to_latex(
        label="tab:descriptive_statistics_rv",
        caption="Stock level summary statistics for realized volatilities",
        float_format="%.3f",
    )
)


##############################################################################################
# Cross sectional summary statistics
##############################################################################################

# Part for RV

descriptive_for_stock_rv = descriptive_for_stock_rv[
    ["Mean", "Std.", "Skew.", "Ex.Kurt"]
]

descriptive_for_aggregated_stock_rv = pd.DataFrame(
    {
        "Min": descriptive_for_stock_rv.min(),
        "0.10": descriptive_for_stock_rv.quantile(0.10),
        "0.25": descriptive_for_stock_rv.quantile(0.25),
        "0.50": descriptive_for_stock_rv.quantile(0.50),
        "0.75": descriptive_for_stock_rv.quantile(0.75),
        "0.90": descriptive_for_stock_rv.quantile(0.90),
        "Max": descriptive_for_stock_rv.max(),
        "Mean": descriptive_for_stock_rv.mean(),
        "Standard Dev.": descriptive_for_stock_rv.std(),
    }
)

descriptive_for_aggregated_stock_rv.rename(index={0: "Stock"}, inplace=True)


## Part for log RV

descriptive_for_stock_log_rv = descriptive_for_stock_log_rv[
    ["Mean", "Std.", "Skew.", "Ex.Kurt"]
]

descriptive_for_aggregated_stock_log_rv = pd.DataFrame(
    {
        "Min": descriptive_for_stock_log_rv.min(),
        "0.10": descriptive_for_stock_log_rv.quantile(0.10),
        "0.25": descriptive_for_stock_log_rv.quantile(0.25),
        "0.50": descriptive_for_stock_log_rv.quantile(0.50),
        "0.75": descriptive_for_stock_log_rv.quantile(0.75),
        "0.90": descriptive_for_stock_log_rv.quantile(0.90),
        "Max": descriptive_for_stock_log_rv.max(),
        "Mean": descriptive_for_stock_log_rv.mean(),
        "Standard Dev.": descriptive_for_stock_log_rv.std(),
    }
)


combined = pd.concat(
    [descriptive_for_aggregated_stock_rv.T, descriptive_for_aggregated_stock_log_rv.T],
    axis=1,
    keys=["Realized Volatility", "Log Realized Volatility"],
)

combined.columns.set_names(["", "Statistic"], inplace=True)


print(
    combined.to_latex(
        caption="Cross-sectional summary statistics for realized volatilities and log realized volatilities for the 25 DJIA stocks.",
        label="tab:descriptives_aggregated",
        float_format="%.3f",
        multicolumn=True,
        multicolumn_format="c",
        escape=False,
    )
)


########################################################################
# Plot raw log rv time series 90 day rolling average
########################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)


df.columns = df.columns.droplevel(0)
tickers = df.columns


fig, axes = plt.subplots(7, 4, figsize=(15, 15), sharex=False, sharey=False)
axes = axes.flatten()

# loop through each ticker
for ax, ticker in zip(axes, tickers):
    ax.plot(df.index, df[ticker].rolling(63).mean(), linewidth=0.8, color="#0000FF")
    ax.set_title(f"{ticker} RV rolling average")
    ax.tick_params(labelsize=6)


plt.tight_layout()
plt.savefig("figures/descriptive_part/rv_ts_plots.png", format="png")
plt.show()


########################################################################
# Plot raw rv time series for 4 stocks main text NO LOG START OF DESCRIPTIVE
########################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

df.columns = df.columns.droplevel(0)
tickers = df.columns
tickers_short = tickers[[6, 8, 13, 20]]
colors = ["#FE9928", "#FF0000", "#008002", "#0000FF"]

fig, ax = plt.subplots(figsize=(15, 7))

for ticker, col in zip(tickers_short, colors):
    ax.plot(
        df.index,
        df[ticker].rolling(window=63).mean(),
        linewidth=1,
        color=col,
        label=ticker,
    )

ax.set_ylabel("Realized Volatility")
ax.legend(title="Ticker")
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("figures/descriptive_part/4_rv_ts_plots.svg", format="svg")
plt.show()


########################################################################
# Plot pacf of log rv
########################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

df = np.log(df)

df.columns = df.columns.droplevel(0)
tickers = df.columns
nlags = 30


fig, axes = plt.subplots(7, 4, figsize=(15, 15), sharex=False, sharey=False)
axes = axes.flatten()

# loop through each ticker
for ax, ticker in zip(axes, tickers):
    plot_pacf(
        df[ticker],
        lags=nlags,
        ax=ax,
        zero=False,
        color="#0000FF",
        vlines_kwargs={"color": "#0000FF"},
    )
    ax.set_title(f"PACF of {ticker}")
    ax.set_ylim(-0.3, 1.0)
    ax.tick_params(labelsize=6)


plt.tight_layout()
plt.savefig("figures/descriptive_part/pacf_log_rv.svg", format="svg")
plt.savefig("figures/descriptive_part/pacf_log_rv.png", format="png")
plt.show()


########################
# Random PACF for main
########################

tickers_short = tickers[[6, 8, 13, 20]]

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.flatten()
nlags = 30

# loop through each ticker
for ax, ticker in zip(axes, tickers_short):
    plot_pacf(
        df[ticker],
        lags=nlags,
        ax=ax,
        zero=False,
        color="#0000FF",
        vlines_kwargs={"color": "#0000FF"},
    )
    ax.set_title(f"PACF of {ticker}")
    ax.set_ylim(-0.3, 1.0)
    ax.tick_params(labelsize=6)


plt.tight_layout()
plt.savefig("figures/descriptive_part/pacf_plot_main.svg", format="svg")
plt.show()


######################################################################################
# Augmented Dickey-Fuller test for log RV
######################################################################################

with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df = pickle.load(f)

df = np.log(df)
df.columns = df.columns.droplevel(0)


results = []
for ticker in df.columns:
    adf_out = adfuller(df[ticker])
    test_stat, p_value, used_lag, n_obs, crit_vals = (
        adf_out[0],
        adf_out[1],
        adf_out[2],
        adf_out[3],
        adf_out[4],
    )
    results.append(
        {
            "Ticker": ticker,
            "ADF Statistic": test_stat,
            "p-value": p_value,
            "Lags": used_lag,
            "Observations": n_obs,
            "1\% Critical Value": crit_vals["1%"],
            "5\% Critical Value": crit_vals["5%"],
        }
    )

# Turn into a DataFrame and latex table
adf_test_results = pd.DataFrame(results).set_index("Ticker")


print(
    adf_test_results.to_latex(
        label="tab:adf_test_log_rv",
        caption="Augmented Dickey-Fuller test for log realized volatilities.",
        float_format="%.3f",
    )
)
