import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader.fred import FredReader
import datetime as dt
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_prod.pkl", mode="rb"
) as f:
    df = pickle.load(f)
# number of daily rv lags for the dnn

# For  calculating rv
with open(
    "/Users/wanja/Developer/BA Thesis code/data/rv_dowj_cleaned.pkl", mode="rb"
) as f:
    df2 = pickle.load(f)

df2.columns = df2.columns.droplevel(0)

########################################################################################


tickers = sorted(df["ticker"].unique())

trading_days = 252
rf_series = "EFFR"

start_date = df["timestamp"].min()
end_date = df["timestamp"].max()

# Download ticker adjusted prices
prices = yf.download(
    tickers, start=start_date, end=end_date, progress=False, auto_adjust=False
)["Adj Close"]

# No missing data
assert prices.isna().sum().sum() == 0, "Missing prices"

# drop first row
log_ret = np.log(prices).diff().dropna()


# Risk free rate from

rf_raw = (
    FredReader(rf_series, start=start_date, end=end_date)
    .read()
    .div(100)
    .rename(columns={rf_series: "rf"})
)


# Forward fill - correct since the old fed fund rate is active till a new replaces it
rf_raw = rf_raw.reindex(log_ret.index).fillna(method="ffill")

# change to daily continuously compounded rate
rf_daily = np.log(1 + rf_raw["rf"]) / 252

# Calcualte excess return calcualte row wise for all 25 stocks
excess = log_ret.sub(rf_daily, axis=0)

# Column wise per stock mean excessreturn
daily_excess_return = excess.mean()

# annualized mean daily rv
annualized_mean_rv = df2.mean() * np.sqrt(252)

# SR based on annualized daily escess return
SR_vec = ((252 * daily_excess_return) / annualized_mean_rv).round(3)

# Compute the correlation matrix based on daily excess returns
Gamma = excess.corr()
Gamma_plot = Gamma
Gamma = Gamma.to_numpy()


# Save both
# with open("/Users/wanja/Developer/BA Thesis code/data/SR_vec.pkl", "wb") as f:
#     pickle.dump(SR_vec, f)

# with open("/Users/wanja/Developer/BA Thesis code/data/Gamma.pkl", "wb") as f:
#     pickle.dump(Gamma, f)

# Visualize

# # Load Gamma and SR
# with open("/Users/wanja/Developer/BA Thesis code/data/SR_vec.pkl", "rb") as f:
#     SR_vec = pickle.load(f)


# with open("/Users/wanja/Developer/BA Thesis code/data/Gamma.pkl", "rb") as f:
#     Gamma = pickle.load(f)


# print(
#     SR_vec.to_frame('SR').to_latex(
#         caption="Sharpe ratios for the 25 DJIA Stocks",
#         label="tab:sharp_ratios",
#         float_format="%.3f",
#     )
# )

# sns.heatmap(Gamma_plot, cmap='coolwarm', center=0)
# plt.savefig("figures/descriptive_part/Gamma.png", format="png")
# plt.show()
