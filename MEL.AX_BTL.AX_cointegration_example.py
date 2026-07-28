import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# =====================================================
# OLS hedge-ratio estimation example: MEL.AX vs BTL.AX
# =====================================================

ticker_x = "MEL.AX"
ticker_y = "BTL.AX"

# Use the first 12-month training window as an illustrative example
start_date = "2022-01-01"
end_date   = "2023-01-01"

# Historical ticker fallback, consistent with the main project logic
ticker_history_map = {
    "BTL.AX": ["BTL.AX", "EEG.AX"]
}

def get_candidate_tickers(ticker):
    return ticker_history_map.get(ticker, [ticker])

# Download all possible tickers
download_tickers = []
for t in [ticker_x, ticker_y]:
    download_tickers.extend(get_candidate_tickers(t))

download_tickers = list(dict.fromkeys(download_tickers))

# -----------------------------------------------------
# 1. Download adjusted close prices
# -----------------------------------------------------
df = yf.download(
    download_tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

# -----------------------------------------------------
# 2. Extract Close prices
# -----------------------------------------------------
if isinstance(df.columns, pd.MultiIndex):
    close = df.xs("Close", axis=1, level=0)
else:
    close = df[["Close"]].copy()

close.index = pd.to_datetime(close.index)
close = close.sort_index()

# -----------------------------------------------------
# 3. Construct final price series with ticker fallback
# -----------------------------------------------------
prices = pd.DataFrame(index=close.index)

for final_ticker in [ticker_x, ticker_y]:
    candidate_tickers = get_candidate_tickers(final_ticker)

    combined_series = None
    used_sources = []

    for raw_ticker in candidate_tickers:
        if raw_ticker in close.columns:
            s = close[raw_ticker].copy()

            if s.dropna().empty:
                continue

            if combined_series is None:
                combined_series = s
            else:
                combined_series = combined_series.combine_first(s)

            used_sources.append(raw_ticker)

    if combined_series is not None and not combined_series.dropna().empty:
        prices[final_ticker] = combined_series
        print(f"{final_ticker} uses data from: {used_sources}")

prices = prices[[ticker_x, ticker_y]].dropna().copy()

# -----------------------------------------------------
# 4. Construct log-price series
# -----------------------------------------------------
X_t = np.log(prices[ticker_x])
Y_t = np.log(prices[ticker_y])

# -----------------------------------------------------
# 5. OLS regression:
#    X_t = alpha + beta Y_t + epsilon_t
# -----------------------------------------------------
Y_with_const = sm.add_constant(Y_t)
ols_model = sm.OLS(X_t, Y_with_const).fit()

alpha_hat = ols_model.params.iloc[0]
beta_hat  = ols_model.params.iloc[1]

# Fitted log-price:
# X_hat_t = alpha_hat + beta_hat Y_t
X_hat_t = alpha_hat + beta_hat * Y_t

# Residual spread:
# S_hat_t = X_t - X_hat_t
spread_hat = X_t - X_hat_t

# -----------------------------------------------------
# 6. Optional statistical tests
# -----------------------------------------------------
coint_stat, coint_p_value, _ = coint(X_t, Y_t)
adf_stat, adf_p_value, _, _, _, _ = adfuller(spread_hat.dropna())

print("====================================")
print(f"Pair: {ticker_x} vs {ticker_y}")
print(f"Training window: {start_date} to {end_date}")
print(f"Estimated alpha: {alpha_hat:.6f}")
print(f"Estimated beta : {beta_hat:.6f}")
print(f"Cointegration p-value: {coint_p_value:.6f}")
print(f"ADF p-value on residual spread: {adf_p_value:.6f}")
print("====================================")

# -----------------------------------------------------
# 7. Plot actual/fitted log-price and residual spread
# -----------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Upper panel: actual X_t and fitted X_hat_t
axes[0].plot(
    X_t.index,
    X_t,
    label=rf"Actual log-price $X_t=\log(P_{{{ticker_x},t}})$",
    linewidth=1.5
)

axes[0].plot(
    X_hat_t.index,
    X_hat_t,
    label=rf"Fitted log-price $\hat{{X}}_t=\hat{{\alpha}}+\hat{{\beta}}Y_t$",
    linewidth=1.5
)

axes[0].set_title(f"OLS Fitted Log-Price: {ticker_x} vs {ticker_y}")
axes[0].set_ylabel("Log-Price")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Lower panel: residual spread
spread_mean = spread_hat.mean()
spread_std = spread_hat.std()

axes[1].plot(
    spread_hat.index,
    spread_hat,
    label=rf"Residual spread $\hat{{S}}_t=X_t-\hat{{X}}_t$",
    linewidth=1.5
)

axes[1].axhline(
    spread_mean,
    linestyle="--",
    linewidth=1.2,
    label="Mean"
)

axes[1].axhline(
    spread_mean + spread_std,
    linestyle=":",
    linewidth=1.2,
    label="+1 Std"
)

axes[1].axhline(
    spread_mean - spread_std,
    linestyle=":",
    linewidth=1.2,
    label="-1 Std"
)

axes[1].set_title("OLS Residual Spread")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Residual Spread")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

# Save figure for Overleaf
plt.savefig("MEL_BTL_OLS_hedge_ratio_example.png", dpi=300, bbox_inches="tight")

plt.show()



# =====================================================
# 8. Classical Z-score baseline strategy illustration
#    Based on the residual spread constructed above
# =====================================================

# Baseline parameters used in the project
LOOKBACK_Z = 20
ENTRY_Z = 2.0
EXIT_Z = 0.5

# -----------------------------------------------------
# 8.1 Rolling Z-score
#     Z_t = (S_t - rolling mean) / rolling std
# -----------------------------------------------------
rolling_mean = spread_hat.rolling(window=LOOKBACK_Z).mean()
rolling_std = spread_hat.rolling(window=LOOKBACK_Z).std()

zscore = (spread_hat - rolling_mean) / rolling_std
zscore = zscore.dropna()

# -----------------------------------------------------
# 8.2 Generate baseline trading positions
#     position =  1: long spread
#     position = -1: short spread
#     position =  0: flat
# -----------------------------------------------------
positions = pd.Series(index=zscore.index, dtype=float)
current_position = 0

for date, z in zscore.items():

    # No position
    if current_position == 0:
        if z > ENTRY_Z:
            current_position = -1      # short spread
        elif z < -ENTRY_Z:
            current_position = 1       # long spread

    # Long spread position
    elif current_position == 1:
        if z > -EXIT_Z:
            current_position = 0       # close long spread

    # Short spread position
    elif current_position == -1:
        if z < EXIT_Z:
            current_position = 0       # close short spread

    positions.loc[date] = current_position

# -----------------------------------------------------
# 8.3 Plot baseline Z-score and position dynamics
# -----------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Upper panel: rolling Z-score with thresholds
axes[0].plot(
    zscore.index,
    zscore,
    label=rf"Rolling Z-score, $L={LOOKBACK_Z}$",
    linewidth=1.5
)

axes[0].axhline(
    ENTRY_Z,
    linestyle="--",
    linewidth=1.2,
    label=rf"Entry threshold $+{ENTRY_Z}$"
)

axes[0].axhline(
    -ENTRY_Z,
    linestyle="--",
    linewidth=1.2,
    label=rf"Entry threshold $-{ENTRY_Z}$"
)

axes[0].axhline(
    EXIT_Z,
    linestyle=":",
    linewidth=1.2,
    label=rf"Exit threshold $+{EXIT_Z}$"
)

axes[0].axhline(
    -EXIT_Z,
    linestyle=":",
    linewidth=1.2,
    label=rf"Exit threshold $-{EXIT_Z}$"
)

axes[0].axhline(
    0,
    linestyle="-",
    linewidth=0.8
)

axes[0].set_title(
    f"Baseline Rolling Z-score Signal: {ticker_x} vs {ticker_y}"
)
axes[0].set_ylabel("Z-score")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Lower panel: trading position
axes[1].step(
    positions.index,
    positions,
    where="post",
    label="Trading position",
    linewidth=1.5
)

axes[1].axhline(
    0,
    linestyle="--",
    linewidth=1.0
)

axes[1].set_yticks([-1, 0, 1])
axes[1].set_yticklabels(["Short spread", "Flat", "Long spread"])

axes[1].set_title("Baseline Position Dynamics")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Position")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

# Save figure for Overleaf
plt.savefig("MEL_BTL_baseline_zscore_example.png", dpi=300, bbox_inches="tight")

plt.show()
