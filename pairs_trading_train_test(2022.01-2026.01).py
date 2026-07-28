import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import random
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import norm


# =========================================
# 1. Parameter settings
# =========================================
N = 10
SEED_LIST = list(range(1,11))
BASELINE_SEED = 2

GLOBAL_START = "2022-01-01"
GLOBAL_END   = "2026-01-01"   # yfinance uess an ecluusive end date, so the data end on 2025-12-31

TRAIN_MONTHS = 12
TEST_MONTHS = 1
STEP_MONTHS = 1

TOP_N_PAIRS_EACH_WINDOW = 5   # select the top five pairs in each rolling window

LOOKBACK_Z = 20
ENTRY_Z = 2.0
EXIT_Z = 0.5
COST_PER_TRADE = 0.001

P_THRESHOLD = 0.05
ADF_THRESHOLD = 0.05
BETA_MIN = 0.05
BETA_MAX = 3.0
MIN_OBS = 100

OUTPUT_PREFIX = "pairs_trading_train_test(2022-2026)"

GAS_TICKERS = [
    "WDS.AX","STO.AX","BPT.AX","AEL.AX","STX.AX","TBN.AX","BTL.AX","CTP.AX","COI.AX","GLL.AX",
    "BRU.AX","VEN.AX","CUE.AX","TDO.AX","MEL.AX","IPB.AX",
    "APA.AX","AGL.AX","ORG.AX","ALD.AX","VEA.AX","EWC.AX",
]

# use the current ticker as the canonical name while also downloading historical tickers
TICKER_HISTORY_MAP = {
    "AEL.AX": ["AEL.AX", "COE.AX"],   # Amplitude Energy, formerly Cooper Energy
    "BTL.AX": ["BTL.AX", "EEG.AX"],   # Beetaloo Energy, formerly Empire Energy
    "IPB.AX": ["IPB.AX", "FEL.AX"],   # IPB.AX remains the appropriate canonical ticker ticker for this sample period
}


all_seed_summary = []
seed2_best_pair_name = None
seed2_prices = None
seed2_test_start_dates = None
seed2_all_test_results_df = None
seed2_top5_pairs_list = None
seed2_top5_test_df = None
seed2_pair_occurrence_df = None
seed2_top5_most_frequent_pairs = None
seed2_portfolio_daily = None
seed2_OUTPUT_PREFIX = None


for SEED in SEED_LIST:
    print(f"\n{'='*40}")
    print(f"Starting run for SEED = {SEED}")
    print(f"{'='*40}")

    seed_dir = Path(f"seed_runs/seed_{SEED}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREFIX = str(seed_dir / f"pairs_trading_seed{SEED}")


    # =========================================
    # 2. Stock universe and data
    # =========================================
    def pick_universe(pool, n=None, seed=None):
        pool = list(dict.fromkeys(pool))
        if not n or n >= len(pool):
            return pool
        if seed is not None:
            random.seed(seed)
        return random.sample(pool, n)



    universe = pick_universe(GAS_TICKERS, N, SEED)
    print(f"Selected stock universe ({len(universe)}): {universe}")

    # =========================================
    # 2a. Construct the download list to handle ticker renaming
    # =========================================
    download_tickers = []

    for t in universe:
        if t in TICKER_HISTORY_MAP:
            download_tickers.extend(TICKER_HISTORY_MAP[t])
        else:
            download_tickers.append(t)

    download_tickers = list(dict.fromkeys(download_tickers))

    print(f" Tickers used for data download ({len(download_tickers)}): {download_tickers}")

    df = yf.download(
        download_tickers,
        start=GLOBAL_START,
        end=GLOBAL_END,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("The downloaded dataset is empty. Please check the ticker symbols or network connection")

    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        close = df[["Close"]] if "Close" in df.columns else df.copy()

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    # =========================================
    # 2b. Constryct the final prices dataset
    # final column names use the canoical tickers in the selected universe
    # For example AEL.AX = AEL.AX data with historical COE.AX  data
    # =========================================
    prices = pd.DataFrame(index=close.index)
    availability_records = []

    for final_ticker in universe:
        candidate_tickers = TICKER_HISTORY_MAP.get(final_ticker, [final_ticker])

        combined_series = None
        used_sources = []

        for raw_ticker in candidate_tickers:
            if raw_ticker in close.columns:
                s = close[raw_ticker]

                if s.dropna().empty:
                    continue

                if combined_series is None:
                    combined_series = s.copy()
                else:
                    combined_series = combined_series.combine_first(s)

                used_sources.append(raw_ticker)

        if combined_series is not None and not combined_series.dropna().empty:
            prices[final_ticker] = combined_series

            valid_s = combined_series.dropna()
            availability_records.append({
                "final_ticker": final_ticker,
                "used_sources": ", ".join(used_sources),
                "first_available_date": valid_s.index.min().date(),
                "last_available_date": valid_s.index.max().date(),
                "n_observations": len(valid_s)
            })

    prices = prices.dropna(axis=1, how="all").copy()
    prices = prices.sort_index()

    availability_df = pd.DataFrame(availability_records)

    print("\n📌 data avaiavukuty check：")
    print(availability_df.to_string(index=False))

    availability_df.to_csv(
        f"{OUTPUT_PREFIX}_ticker_data_availability.csv",
        index=False
    )

    print(f"\nfinal valid stocks ({len(prices.columns)}): {list(prices.columns)}")

    removed = [x for x in universe if x not in prices.columns]
    print(f"removed stocks ({len(removed)}): {removed}")

    print(f"price data range: {prices.index.min().date()} ~ {prices.index.max().date()}")

    prices.to_csv(f"{OUTPUT_PREFIX}_prices.csv")
    print(f"✅ price data save to: {OUTPUT_PREFIX}_prices.csv")
    print(f"✅ Ticker data avilability report saved to: {OUTPUT_PREFIX}_ticker_data_availability.csv")


    # =========================================
    # 2c. Diagnositic check for abnormal price jumps
    # =========================================
    price_ret_check = prices.pct_change()

    abnormal_mask = price_ret_check.abs() > 0.5
    abnormal_stacked = price_ret_check[abnormal_mask].stack()
    abnormal_df = abnormal_stacked.rename("daily_return").reset_index()
    abnormal_df.columns = ["date", "ticker", "daily_return"]
    abnormal_df = abnormal_df.sort_values("daily_return", key=abs, ascending=False)

    print(f"\n⚠️ [SEED={SEED}] number of observations with absolute daily returns above 50%：{len(abnormal_df)}")
    if len(abnormal_df) > 0:
        print(abnormal_df.to_string(index=False))

    abnormal_df.to_csv(f"{OUTPUT_PREFIX}_abnormal_daily_returns.csv", index=False)

    print("\n📌 price check around the largest jump for merged ticker seies：")
    for final_ticker in TICKER_HISTORY_MAP.keys():
        if final_ticker not in prices.columns:
            continue
        s = prices[final_ticker].dropna()
        if len(s) < 5:
            continue
        ret_series = s.pct_change().dropna()
        max_jump = ret_series.abs().max()
        max_jump_date = ret_series.abs().idxmax()
        print(f"{final_ticker}: Largest absolute daily return = {max_jump:.2%}，observed on {max_jump_date.date()}")
        window_start = max_jump_date - pd.Timedelta(days=7)
        window_end = max_jump_date + pd.Timedelta(days=7)
        nearby = s[(s.index >= window_start) & (s.index <= window_end)]
        print(nearby.to_string())
        print("-" * 40)




    # =========================================
    # 3. utility functions
    # =========================================
    def compute_spread(price1, price2, beta, alpha=0.0):
        log_p1 = np.log(price1)
        log_p2 = np.log(price2)
        return log_p1 - alpha - beta * log_p2
    

    def rolling_zscore(series, window=20):
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std

    def generate_positions(zscore, entry_z=2.0, exit_z=0.5):
        position = pd.Series(index=zscore.index, dtype=float)
        current_pos = 0

        for i in range(len(zscore)):
            z = zscore.iloc[i]

            if pd.isna(z):
                position.iloc[i] = current_pos
                continue

            # exit the exiting position
            if current_pos == 1 and z > -exit_z:
                current_pos = 0
            elif current_pos == -1 and z < exit_z:
                current_pos = 0

            # open a new position
            if current_pos == 0:
                if z < -entry_z:
                    current_pos = 1      # long spread
                elif z > entry_z:
                    current_pos = -1     # short spread

            position.iloc[i] = current_pos

        return position

    
    def backtest_pair(
            price1,
            price2,
            beta,
            alpha=0.0,
            lookback_z=20,
            entry_z=2.0,
            exit_z=0.5,
            cost_per_trade=0.0,
            trade_start=None,
            force_close_at_end=False
            ):
        df_bt = pd.DataFrame(index=price1.index)
        df_bt["price1"] = price1
        df_bt["price2"] = price2
        df_bt = df_bt.dropna().copy()

        # Daily simple returns
        df_bt["r1"] = df_bt["price1"].pct_change()
        df_bt["r2"] = df_bt["price2"].pct_change()

        # Residual spread and rolling Z-score
        df_bt["spread"] = compute_spread(
            df_bt["price1"],
            df_bt["price2"],
            beta=beta,
            alpha=alpha
        )

        df_bt["zscore"] = rolling_zscore(
            df_bt["spread"],
            window=lookback_z
        )

        # -----------------------------------------
        # Warm-up observations only initialise Z-score.
        # Actual positions start from trade_start.
        # -----------------------------------------
        if trade_start is None:
            df_bt["position"] = generate_positions(
                df_bt["zscore"],
                entry_z=entry_z,
                exit_z=exit_z
            )
        else:
            trade_start = pd.Timestamp(trade_start)

            df_bt["position"] = 0.0

            live_zscore = df_bt.loc[
                df_bt.index >= trade_start,
                "zscore"
            ]

            if not live_zscore.empty:
                live_position = generate_positions(
                    live_zscore,
                    entry_z=entry_z,
                    exit_z=exit_z
                )

                df_bt.loc[live_position.index, "position"] = live_position

        # If each monthly testing window is treated independently,
        # close any remaining position at the final testing observation.
        if force_close_at_end:
            if trade_start is None:
                live_index = df_bt.index
            else:
                live_index = df_bt.index[df_bt.index >= trade_start]

            if len(live_index) > 0:
                final_index = live_index[-1]
                df_bt.loc[final_index, "position"] = 0.0

        df_bt["position_lag"] = df_bt["position"].shift(1).fillna(0.0)

        # Spread return under the existing notional convention
        df_bt["spread_ret"] = (df_bt["r1"] - beta * df_bt["r2"]) / (1 + abs(beta))

        # Direction diagnostic
        df_bt["predicted_direction"] = df_bt["position_lag"]
        df_bt["actual_direction"] = (
            np.sign(df_bt["spread_ret"]).replace(0, np.nan)
        )

        df_bt["direction_hit"] = np.where(
            df_bt["predicted_direction"] == 0,
            np.nan,
            (
                np.sign(df_bt["predicted_direction"])
                == np.sign(df_bt["spread_ret"])
            ).astype(float)
        )

        df_bt["cum_hit_rate"] = (
            df_bt["direction_hit"]
            .expanding(min_periods=1)
            .mean()
        )

        # Gross strategy return: lagged position avoids look-ahead bias
        df_bt["strategy_ret_gross"] = (
            df_bt["position_lag"] * df_bt["spread_ret"]
        )

        # -----------------------------------------
        # Position changes and completed trades
        # -----------------------------------------
        previous_position = df_bt["position"].shift(1).fillna(0.0)
        position_changed = df_bt["position"].ne(previous_position)

        # Opening includes flat -> nonzero and direct reversal
        df_bt["open_event"] = (
            df_bt["position"].ne(0) & position_changed
        ).astype(int)

        # Closing includes nonzero -> flat and direct reversal
        df_bt["close_event"] = (
            previous_position.ne(0) & position_changed
        ).astype(int)

        df_bt["turnover"] = (
            df_bt["position"] - previous_position
        ).abs()

        # Empirical return-level transaction cost

        df_bt["cost"] = (
            df_bt["turnover"]
            * cost_per_trade
        )

        df_bt["strategy_ret_net"] = (
            df_bt["strategy_ret_gross"] - df_bt["cost"]
        )

        df_bt["cum_gross"] = (
            1 + df_bt["strategy_ret_gross"].fillna(0)
        ).cumprod()

        df_bt["cum_net"] = (
            1 + df_bt["strategy_ret_net"].fillna(0)
        ).cumprod()

        df_bt["predicted_cum_direction"] = (
            df_bt["predicted_direction"].fillna(0).cumsum()
        )

        df_bt["actual_cum_spread_ret"] = (
            df_bt["spread_ret"].fillna(0).cumsum()
        )

        return df_bt


    def calc_metrics(ret_series):
        ret = ret_series.dropna()
        if len(ret) == 0:
            return {
                "total_return": np.nan,
                "annual_return": np.nan,
                "annual_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan
            }

        cum = (1 + ret).cumprod()
        total_return = cum.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(ret)) - 1
        daily_mean = ret.mean()
        daily_std = ret.std(ddof=1)
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = (
            np.sqrt(252)*daily_mean/daily_std 
         if daily_std > 0 else np.nan
         )

        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        }

    def evaluate_pair_on_train(train_data, s1, s2,
                            coint_p_threshold=0.05,
                            adf_p_threshold=0.05,
                            min_obs=100,
                            beta_min=0.05,
                            beta_max=3.0):
        if s1 not in train_data.columns or s2 not in train_data.columns:
            return None

        pair_data = train_data[[s1, s2]].dropna().copy()
        if len(pair_data) < min_obs:
            return None

        log_p1 = np.log(pair_data[s1])
        log_p2 = np.log(pair_data[s2])

        try:
            model = sm.OLS(log_p1, sm.add_constant(log_p2)).fit()
            alpha = model.params.iloc[0]
            beta = model.params.iloc[1]

            if abs(beta) < beta_min or abs(beta) > beta_max:
                return None

            coint_stat, coint_p, _ = coint(log_p1, log_p2)

            spread = log_p1 - alpha - beta * log_p2
            adf_p = adfuller(spread.dropna())[1]

            if coint_p >= coint_p_threshold:
                return None

            if adf_p >= adf_p_threshold:
                return None

            return {
                "pair": f"{s1} vs {s2}",
                "s1": s1,
                "s2": s2,
                "beta": beta,
                "alpha": alpha,
                "train_coint_p_value": coint_p,
                "train_adf_p_value": adf_p,
                "rank_score": (coint_p, adf_p)
            }

        except Exception as e:
            return None

    def period_return_from_daily(net_ret_series):
        net_ret_series = net_ret_series.dropna()
        if len(net_ret_series) == 0:
            return np.nan
        return (1 + net_ret_series).prod() - 1

    # =========================================
    # 4. generate out-of-sample test start dates
    # =========================================
    test_start_dates = []
    current_test_start = pd.Timestamp(GLOBAL_START) + pd.DateOffset(months=TRAIN_MONTHS)
    global_end_ts = pd.Timestamp(GLOBAL_END)

    while current_test_start + pd.DateOffset(months=TEST_MONTHS) <= global_end_ts:
        test_start_dates.append(current_test_start)
        current_test_start = current_test_start + pd.DateOffset(months=STEP_MONTHS)

    print(f"✅ Number of rolling Train-Test windows: {len(test_start_dates)}")

    # =========================================
    # 5. Rolling training and out-of-sample testing
    # =========================================
    all_monthly_selected_pairs = []
    all_test_results = []

    for test_start in test_start_dates:
        train_end = test_start
        train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)

        train_data = prices[(prices.index >= train_start) & (prices.index < train_end)].copy()
        test_data  = prices[(prices.index >= test_start) & (prices.index < test_end)].copy()

        if train_data.empty or test_data.empty:
            continue

        print(f"\n==============================")
        print(f"current training window: {train_start.date()} ~ {(train_end - pd.Timedelta(days=1)).date()}")
        print(f"current testing window: {test_start.date()} ~ {(test_end - pd.Timedelta(days=1)).date()}")
        print(f"==============================")

        train_pairs_eval = []
        pairs = list(itertools.combinations(train_data.columns, 2))

        for s1, s2 in pairs:
            est = evaluate_pair_on_train(
                train_data,
                s1, s2,
                coint_p_threshold=P_THRESHOLD,
                adf_p_threshold=ADF_THRESHOLD,
                min_obs=MIN_OBS,
                beta_min=BETA_MIN,
                beta_max=BETA_MAX
            )
            if est is not None:
                train_pairs_eval.append(est)

        if len(train_pairs_eval) == 0:
            print("⚠️ No pair passed the training-stage screening in this window")
            continue

        train_pairs_df = pd.DataFrame(train_pairs_eval).sort_values(
            ["train_coint_p_value", "train_adf_p_value"],
            ascending=[True, True]
        ).reset_index(drop=True)

        top_pairs_df = train_pairs_df.head(TOP_N_PAIRS_EACH_WINDOW).copy()
        top_pairs_df["train_start"] = train_start.date()
        top_pairs_df["train_end"] = (train_end - pd.Timedelta(days=1)).date()
        top_pairs_df["test_start"] = test_start.date()
        top_pairs_df["test_end"] = (test_end - pd.Timedelta(days=1)).date()
        top_pairs_df["rank_in_window"] = np.arange(1, len(top_pairs_df) + 1)

        all_monthly_selected_pairs.append(top_pairs_df)

        for _, row in top_pairs_df.iterrows():
            pair_name = row["pair"]
            s1 = row["s1"]
            s2 = row["s2"]
            beta = row["beta"]
            alpha = row["alpha"]

            if s1 not in test_data.columns or s2 not in test_data.columns:
                continue

            # warm-up：ensure the z-score canbe calculated at the beginning of the testing month
            warmup_data = train_data[[s1, s2]].dropna().tail(LOOKBACK_Z + 1)
            combined_data = pd.concat([warmup_data, test_data[[s1, s2]].dropna()], axis=0)

            result_full = backtest_pair(
                price1=combined_data[s1],
                price2=combined_data[s2],
                beta=beta,
                alpha=alpha,
                lookback_z=LOOKBACK_Z,
                entry_z=ENTRY_Z,
                exit_z=EXIT_Z,
                cost_per_trade=COST_PER_TRADE,
                trade_start=test_start,
                force_close_at_end=True
            )

            result_test = result_full[result_full.index >= test_start].copy()
            if result_test.empty:
                continue

            result_test["pair"] = pair_name
            result_test["beta"] = beta
            result_test["alpha"] = alpha
            result_test["train_start"] = train_start.date()
            result_test["train_end"] = (train_end - pd.Timedelta(days=1)).date()
            result_test["test_start"] = test_start.date()
            result_test["test_end"] = (test_end - pd.Timedelta(days=1)).date()
            result_test["rank_in_window"] = int(row["rank_in_window"])

            all_test_results.append(result_test)

    # =========================================
    # 6. Summary: Monthly Selected Pairs and the five most frequently selected pairs
    # =========================================
    if len(all_monthly_selected_pairs) == 0:
        raise ValueError("No eligible pair wa selected in any rolling window. Please check the screening criteria。")

    monthly_selected_pairs_df = pd.concat(all_monthly_selected_pairs, axis=0).reset_index(drop=True)
    monthly_selected_pairs_df.to_csv(f"{OUTPUT_PREFIX}_monthly_selected_top5_pairs.csv", index=False)
    print(f"✅ saved：{OUTPUT_PREFIX}_monthly_selected_top5_pairs.csv")

    pair_occurrence_df = (
        monthly_selected_pairs_df.groupby("pair")
        .agg(
            occurrence_count=("pair", "count"),
            avg_train_coint_p_value=("train_coint_p_value", "mean"),
            avg_train_adf_p_value=("train_adf_p_value", "mean"),
            avg_beta=("beta", "mean")
        )
        .reset_index()
        .sort_values(
            ["occurrence_count", "avg_train_coint_p_value", "avg_train_adf_p_value"],
            ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )

    top5_most_frequent_pairs = pair_occurrence_df.head(5).copy()
    top5_most_frequent_pairs.to_csv(f"{OUTPUT_PREFIX}_top5_most_frequent_pairs.csv", index=False)

    print("\n📊 Five most frequently selected pairs：")
    print(top5_most_frequent_pairs)
    print(f"✅ saved：{OUTPUT_PREFIX}_top5_most_frequent_pairs.csv")

    # =========================================
    # 7. Aggregate out-of-sample testing results
    # =========================================
    if len(all_test_results) == 0:
        raise ValueError("No pair produced valid out-of sample test results. Please check the screening critesia.")

    all_test_results_df = pd.concat(all_test_results, axis=0).sort_index()
    all_test_results_df.to_csv(f"{OUTPUT_PREFIX}_all_test_results_daily.csv", index=True)
    print(f"✅ saved：{OUTPUT_PREFIX}_all_test_results_daily.csv")

    top5_pairs_list = top5_most_frequent_pairs["pair"].tolist()
    top5_test_df = all_test_results_df[all_test_results_df["pair"].isin(top5_pairs_list)].copy()


    full_oos_index = prices[
    (prices.index >= min(test_start_dates)) &
    (prices.index < pd.Timestamp(GLOBAL_END))
    ].index

    # =========================================
    # 8a. annual and overall returns for each pair
    # =========================================
    annual_pair_returns = []
    overall_pair_metrics = []

    for pair_name in top5_pairs_list:
        pair_df = top5_test_df[
            top5_test_df["pair"] == pair_name
        ].copy()

        if pair_df.empty:
            continue

        # A pair can appear at most once on each date.
        # If it is not selected in a rolling window,
        # its strategy return is set to zero.
        pair_ret_full = (
            pair_df
            .groupby(level=0)["strategy_ret_net"]
            .sum()
            .reindex(full_oos_index, fill_value=0.0)
            .sort_index()
        )

        annual_ret = (
            pair_ret_full
            .groupby(pair_ret_full.index.year)
            .apply(period_return_from_daily)
            .rename_axis("year")
            .reset_index(name="annual_net_return")
        )

        annual_ret["pair"] = pair_name
        annual_pair_returns.append(annual_ret)

        metrics = calc_metrics(pair_ret_full)
        metrics["pair"] = pair_name

        metrics["occurrence_count"] = int(
            pair_occurrence_df.loc[
                pair_occurrence_df["pair"] == pair_name,
                "occurrence_count"
            ].iloc[0]
        )

        overall_pair_metrics.append(metrics)
    

    if len(annual_pair_returns) > 0:
        annual_pair_returns_df = pd.concat(annual_pair_returns, axis=0).reset_index(drop=True)
    else:
        annual_pair_returns_df = pd.DataFrame(columns=["year", "annual_net_return", "pair"])

    annual_pair_returns_df = annual_pair_returns_df[["pair", "year", "annual_net_return"]]
    annual_pair_returns_df.to_csv(f"{OUTPUT_PREFIX}_top5_pairs_annual_returns.csv", index=False)
    print(f"✅ saved：{OUTPUT_PREFIX}_top5_pairs_annual_returns.csv")

    overall_pair_metrics_df = pd.DataFrame(overall_pair_metrics).sort_values("sharpe", ascending=False)
    overall_pair_metrics_df.to_csv(f"{OUTPUT_PREFIX}_top5_pairs_total_metrics.csv", index=False)
    print(f"✅ saved：{OUTPUT_PREFIX}_top5_pairs_total_metrics.csv")


    # =========================================
    # 8b. Selected the best pair from the top five using the sharpe ratio
    # =========================================

    # add trade counts
    for pair_name in top5_pairs_list:
        pair_df = top5_test_df[top5_test_df["pair"] == pair_name].copy()
        trade_count = int(pair_df["close_event"].gt(0).sum())
        overall_pair_metrics_df.loc[
            overall_pair_metrics_df["pair"] == pair_name, "trade_count"
        ] = trade_count

    # hard diltering criteria
    MIN_TRADE_COUNT    = 5
    MIN_TOTAL_RETURN   = 0.0
    MAX_DRAWDOWN_LIMIT = -0.3

    filtered_df = overall_pair_metrics_df[
        (overall_pair_metrics_df["trade_count"]  >= MIN_TRADE_COUNT)  &
        (overall_pair_metrics_df["total_return"] >  MIN_TOTAL_RETURN) &
        (overall_pair_metrics_df["max_drawdown"] >= MAX_DRAWDOWN_LIMIT) &
        (overall_pair_metrics_df["sharpe"].notna())
    ].copy()

    if filtered_df.empty:
        print("⚠️ no pair among the top5 passed the hard filters. Please relax the criteria")
    else:
        # sort by Sharpe ratio in descending order
        filtered_df = filtered_df.sort_values(
            ["sharpe", "total_return", "max_drawdown", "trade_count"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        filtered_df["selection_score"] = filtered_df["sharpe"]

        filtered_df.to_csv(
            f"{OUTPUT_PREFIX}_top5_pairs_sharpe_selection.csv",
            index=False
        )

        print(f"✅ save：{OUTPUT_PREFIX}_top5_pairs_sharpe_selection.csv")

        best_pair_name = filtered_df.iloc[0]["pair"]

        print(f"\n🏆 Best pair selected by Sharpe ratio: {best_pair_name}")
        print(
            filtered_df[
                ["pair", "sharpe", "total_return", "max_drawdown",
                "trade_count", "occurrence_count"]
            ].to_string(index=False)
        )
        
     
 
    # =========================================
    # 8c：Store the key variables for seed 2 for use in section9 and subsequent plots
    # =========================================
    if filtered_df.empty:
        all_seed_summary.append({
            "seed": SEED,
            "universe": ",".join(universe),
            "best_pair": None,
            "sharpe": np.nan,
            "total_return": np.nan,
            "max_drawdown": np.nan,
            "trade_count": np.nan,
            "occurrence_count": np.nan,
            "selection_score": np.nan
        })
    else:
        best_pair_name = filtered_df.iloc[0]["pair"]
        all_seed_summary.append({
            "seed": SEED,
            "universe": ",".join(universe),
            "best_pair": best_pair_name,
            "sharpe": filtered_df.iloc[0]["sharpe"],
            "total_return": filtered_df.iloc[0]["total_return"],
            "max_drawdown": filtered_df.iloc[0]["max_drawdown"],
            "trade_count": filtered_df.iloc[0]["trade_count"],
            "occurrence_count": filtered_df.iloc[0]["occurrence_count"],
            "selection_score": filtered_df.iloc[0]["selection_score"]
        })

        # store the key variables for seed 2 for use in sectiion 9 and subsequent plots
        if SEED == BASELINE_SEED:
            seed2_best_pair_name = best_pair_name
            seed2_prices = prices.copy()
            seed2_test_start_dates = test_start_dates.copy()
            seed2_all_test_results_df = all_test_results_df.copy()
            seed2_top5_pairs_list = top5_pairs_list.copy()
            seed2_top5_test_df = top5_test_df.copy()
            seed2_pair_occurrence_df = pair_occurrence_df.copy()
            seed2_top5_most_frequent_pairs = top5_most_frequent_pairs.copy()
            seed2_OUTPUT_PREFIX = str(seed_dir / f"pairs_trading_seed{SEED}")

# =========================================
# After completing the seed loop: cross-seed summary
# =========================================
from pathlib import Path

seed_summary_df = pd.DataFrame(all_seed_summary)
seed_summary_df.to_csv("seed_runs/seed_robustness_summary.csv", index=False)

print("\n" + "="*50)
print("📊 cross-seed robusteness summary：")
print("="*50)
print(seed_summary_df.to_string(index=False))

# robbustness statistics
valid_seeds = seed_summary_df.dropna(subset=["sharpe"])
print(f"\n number of valid seeds with a selected best pair: {len(valid_seeds)} / {len(SEED_LIST)}")
print(f"Sharpe mean: {valid_seeds['sharpe'].mean():.4f}")
print(f"Sharpe standard deviation: {valid_seeds['sharpe'].std():.4f}")
print(f"Sharpe minimum: {valid_seeds['sharpe'].min():.4f}")
print(f"Sharpe maximum: {valid_seeds['sharpe'].max():.4f}")
print(f"proportion of seed with positive total returns: {(valid_seeds['total_return'] > 0).mean():.1%}")
print(f"\n✅ save：seed_runs/seed_robustness_summary.csv")
        





# =========================================
# 9. Z-score Parameter sensitiviy analysis：
# Compare static full-period and rolling calibrated
# =========================================

if seed2_best_pair_name is None:
    raise ValueError("no valid best_pair wa generated for seed=2 ，please check.")

#Restore variables asscoiated with seed=2
best_pair_name = seed2_best_pair_name
prices = seed2_prices
test_start_dates = seed2_test_start_dates
all_test_results_df = seed2_all_test_results_df
top5_pairs_list = seed2_top5_pairs_list
top5_test_df = seed2_top5_test_df
pair_occurrence_df = seed2_pair_occurrence_df
top5_most_frequent_pairs = seed2_top5_most_frequent_pairs.copy()
OUTPUT_PREFIX = seed2_OUTPUT_PREFIX

print(f"9  seed=2 best_pair: {best_pair_name}")

best_s1, best_s2 = best_pair_name.split(" vs ")


if "best_pair_name" not in globals():
    raise ValueError("best_pair_name ia undefined. Please execute section 8b first")

best_s1, best_s2 = best_pair_name.split(" vs ")

print("\n==============================")
print("Starting z-score parameter sensitivity analysis")
print(f"Best pair: {best_pair_name}")
print("==============================")

# parameter grid
LOOKBACK_GRID = [10, 20, 30, 40]
ENTRY_Z_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
EXIT_Z_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 2.0]
# =========================================
# Normal-theory benchmark thresholds
# =========================================

def normal_opportunity_equation(s):
    """
    First-order condition of the normal-theory opportunity function:

        J_N(s) = s * P(|Z| > s)
               = 2s[1 - Phi(s)].

    Differentiating with respect to s gives:

        1 - Phi(s) - s * phi(s) = 0.
    """
    return (
        1.0
        - norm.cdf(s)
        - s * norm.pdf(s)
    )


def solve_cost_adjusted_normal_threshold(
    lambda_rt,
    initial_guess=None,
    tolerance=1e-10,
    max_iterations=100
):
    """
    Solve the cost-adjusted normal-theory first-order condition
    using the Newton--Raphson method:

        g(s)
        = 1 - Phi(s)
          - (s - lambda_rt) * phi(s)
        = 0,

    subject to:

        s > lambda_rt.

    The derivative is

        g'(s)
        = phi(s) * [s(s - lambda_rt) - 2].

    Parameters
    ----------
    lambda_rt : float
        Round-trip transaction cost expressed in Z-score units.

    initial_guess : float or None
        Initial Newton--Raphson value. If None, a feasible initial
        value is selected automatically.

    tolerance : float
        Convergence tolerance.

    max_iterations : int
        Maximum number of Newton iterations.

    Returns
    -------
    float
        Cost-adjusted normal-theory threshold. Returns NaN if
        the iteration fails.
    """

    if not np.isfinite(lambda_rt) or lambda_rt < 0:
        return np.nan

    # Initial value must satisfy s > lambda_rt.
    if initial_guess is None:
        s_current = max(0.75, lambda_rt + 0.10)
    else:
        s_current = float(initial_guess)

    if not np.isfinite(s_current):
        return np.nan

    if s_current <= lambda_rt:
        s_current = lambda_rt + 0.10

    for _ in range(max_iterations):

        phi_s = norm.pdf(s_current)

        g_value = (
            1.0
            - norm.cdf(s_current)
            - (s_current - lambda_rt) * phi_s
        )

        g_derivative = (
            phi_s
            * (
                s_current
                * (s_current - lambda_rt)
                - 2.0
            )
        )

        if (
            not np.isfinite(g_value)
            or not np.isfinite(g_derivative)
            or abs(g_derivative) < 1e-14
        ):
            return np.nan

        s_next = (
            s_current
            - g_value / g_derivative
        )

        # Preserve the admissible condition s > lambda_rt.
        if not np.isfinite(s_next):
            return np.nan

        if s_next <= lambda_rt:
            s_next = (
                s_current + lambda_rt
            ) / 2.0

            if s_next <= lambda_rt:
                s_next = lambda_rt + 1e-8

        if abs(s_next - s_current) < tolerance:
            return float(s_next)

        s_current = s_next

    return np.nan


# Cost-free normal-theory opportunity threshold
NORMAL_OPPORTUNITY_THRESHOLD = (
    solve_cost_adjusted_normal_threshold(
        lambda_rt=0.0,
        initial_guess=0.75
    )
)


# Conventional two-sided 5% normal threshold
NORMAL_RARE_THRESHOLD_5PCT = norm.ppf(0.975)

print("\n📌 Normal-theory benchmark thresholds:")
print(
    "Opportunity-maximising threshold: "
    f"{NORMAL_OPPORTUNITY_THRESHOLD:.4f}"
)
print(
    "Two-sided 5% rare-deviation threshold: "
    f"{NORMAL_RARE_THRESHOLD_5PCT:.4f}"
)
print(
    "Conventional baseline entry threshold: "
    f"{ENTRY_Z:.4f}"
)


# Testing period: start from the first out-of-sample test month 
first_test_start = min(test_start_dates)
final_test_end = pd.Timestamp(GLOBAL_END)

sensitivity_summary = []
sensitivity_daily_results = []


# -----------------------------------------
# Utility function: estimate the hedge ratio using OLS
# -----------------------------------------
def estimate_alpha_beta_ols(data, s1, s2):
    pair_data = data[[s1, s2]].dropna().copy()

    if len(pair_data) < MIN_OBS:
        return np.nan, np.nan

    log_p1 = np.log(pair_data[s1])
    log_p2 = np.log(pair_data[s2])

    try:
        model = sm.OLS(log_p1, sm.add_constant(log_p2)).fit()
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]
        return alpha, beta
    except Exception:
        return np.nan, np.nan

# =========================================
# 9a. Project-specific theoretical threshold
#construct the theoretical opportunity score using the empirical tail probability of the training z-score
# =========================================

def compute_theoretical_threshold_scores(
    train_data,
    s1,
    s2,
    lookback_z,
    entry_grid,
    exit_grid,
    cost_per_trade
):
    """
    Project-specific theoretical opportunity scores computed
    using training-sample information only.

    Gross score:
        J_gross
        = (entry - exit)
          * P_train(|Z_t| > entry)

    Cost-adjusted score:
        J_net
        = max(entry - exit - lambda_rt, 0)
          * P_train(|Z_t| > entry)

    where:
        lambda_rt
        = 2c(1 + |beta|) / sigma_hat_{S,L}.
    """

    alpha, beta = estimate_alpha_beta_ols(
        train_data,
        s1,
        s2
    )

    if pd.isna(beta):
        return pd.DataFrame()

    pair_data = train_data[[s1, s2]].dropna().copy()

    if len(pair_data) <= lookback_z + 5:
        return pd.DataFrame()

    spread = compute_spread(
        pair_data[s1],
        pair_data[s2],
        beta=beta,
        alpha=alpha
    )

    rolling_mean = spread.rolling(lookback_z).mean()
    rolling_std = spread.rolling(lookback_z).std()

    z_train = (
        (spread - rolling_mean) / rolling_std
    ).replace([np.inf, -np.inf], np.nan).dropna()

    valid_rolling_std = (
        rolling_std
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(z_train) == 0 or len(valid_rolling_std) == 0:
        return pd.DataFrame()

   
    sigma_hat_s_l = valid_rolling_std.median()

    if (
        not np.isfinite(sigma_hat_s_l)
        or sigma_hat_s_l <= 0
    ):
        return pd.DataFrame()

    # Approximate two-leg round-trip transaction cost
    round_trip_cost = (
        2.0
        * cost_per_trade
        * (1.0 + abs(beta))
    )

    # Convert return-level cost into Z-score units
    lambda_rt = round_trip_cost / sigma_hat_s_l

    # Cost-adjusted stylised normal benchmark
    normal_cost_adjusted_threshold = (
        solve_cost_adjusted_normal_threshold(lambda_rt)
    )

    abs_z = z_train.abs()
    rows = []

    for entry_z in entry_grid:
        empirical_tail_prob = (
            abs_z > entry_z
        ).mean()

        for exit_z in exit_grid:
            if exit_z >= entry_z:
                continue

            gross_gap = entry_z - exit_z

            net_gap = max(
                gross_gap - lambda_rt,
                0.0
            )

            gross_theoretical_score = (
                gross_gap * empirical_tail_prob
            )

            net_theoretical_score = (
                net_gap * empirical_tail_prob
            )

            rows.append({
                "lookback_z": lookback_z,
                "entry_z": entry_z,
                "exit_z": exit_z,

                "gross_gap": gross_gap,
                "round_trip_cost": round_trip_cost,
                "sigma_hat_s_l": sigma_hat_s_l,
                "lambda_rt": lambda_rt,
                "net_gap": net_gap,

                "empirical_tail_prob": empirical_tail_prob,

                "gross_theoretical_score":
                    gross_theoretical_score,

                "net_theoretical_score":
                    net_theoretical_score,

                "normal_cost_adjusted_threshold":
                    normal_cost_adjusted_threshold,

                "alpha": alpha,
                "beta": beta,
                "n_z_obs": len(z_train)
            })

    return pd.DataFrame(rows)



# =========================================
# 9a-i. Static project-specific theoretical threshold
# =========================================

static_train_start = pd.Timestamp(GLOBAL_START)
static_train_end = first_test_start

static_train_data = prices[
    (prices.index >= static_train_start) &
    (prices.index < static_train_end)
].copy()

static_theory_scores = []

for lookback_z_value in LOOKBACK_GRID:
    score_df = compute_theoretical_threshold_scores(
        train_data=static_train_data,
        s1=best_s1,
        s2=best_s2,
        lookback_z=lookback_z_value,
        entry_grid=ENTRY_Z_GRID,
        exit_grid=EXIT_Z_GRID,
        cost_per_trade=COST_PER_TRADE
    )

    if not score_df.empty:
        score_df["mode"] = "static_theoretical"
        score_df["train_start"] = static_train_start.date()
        score_df["train_end"] = (static_train_end - pd.Timedelta(days=1)).date()
        static_theory_scores.append(score_df)

if len(static_theory_scores) > 0:
    static_theory_df = pd.concat(static_theory_scores, axis=0).reset_index(drop=True)

    static_theory_best = (
        static_theory_df
        .sort_values(
            ["net_theoretical_score", "gross_theoretical_score","empirical_tail_prob"],
            ascending=[False, False, False]
        )
        .head(1)
        .reset_index(drop=True)
    )

    static_theory_df.to_csv(
        f"{OUTPUT_PREFIX}_project_theoretical_threshold_static_all_scores.csv",
        index=False
    )

    static_theory_best.to_csv(
        f"{OUTPUT_PREFIX}_project_theoretical_threshold_static_best.csv",
        index=False
    )

    print("\n📌 Static theoretical threshold:")
    print(static_theory_best[
    [
        "lookback_z",
        "entry_z",
        "exit_z",
        "gross_gap",
        "lambda_rt",
        "net_gap",
        "empirical_tail_prob",
        "gross_theoretical_score",
        "net_theoretical_score",
        "normal_cost_adjusted_threshold"
    ]
    ].to_string(index=False))
else:
    print("⚠️ Static theoretical threshold could not be computed.")




# =========================================
# 9a-ii. Rolling project-specific theoretical threshold
# 每个 rolling training window 估计一次 theoretical threshold
# =========================================

rolling_theory_scores = []

for lookback_z_value in LOOKBACK_GRID:
    for test_start in test_start_dates:
        train_end = test_start
        train_start = train_end - pd.DateOffset(months=TRAIN_MONTHS)

        train_data = prices[
            (prices.index >= train_start) &
            (prices.index < train_end)
        ].copy()

        if train_data.empty:
            continue

        if best_s1 not in train_data.columns or best_s2 not in train_data.columns:
            continue

        score_df = compute_theoretical_threshold_scores(
            train_data=train_data,
            s1=best_s1,
            s2=best_s2,
            lookback_z=lookback_z_value,
            entry_grid=ENTRY_Z_GRID,
            exit_grid=EXIT_Z_GRID,
            cost_per_trade=COST_PER_TRADE
        )

        if score_df.empty:
            continue

        score_df["mode"] = "rolling_theoretical"
        score_df["train_start"] = train_start.date()
        score_df["train_end"] = (train_end - pd.Timedelta(days=1)).date()
        score_df["test_start"] = test_start.date()

        rolling_theory_scores.append(score_df)

if len(rolling_theory_scores) > 0:
    rolling_theory_df = pd.concat(rolling_theory_scores, axis=0).reset_index(drop=True)

    rolling_theory_df.to_csv(
        f"{OUTPUT_PREFIX}_project_theoretical_threshold_rolling_all_scores.csv",
        index=False
    )

    # Method 1: select the best average of theoretical score form rolling training windows 
    rolling_theory_summary = (
    rolling_theory_df
    .groupby(
        ["lookback_z", "entry_z", "exit_z"],
        as_index=False
    )
    .agg(
        avg_net_theoretical_score=(
            "net_theoretical_score",
            "mean"
        ),
        median_net_theoretical_score=(
            "net_theoretical_score",
            "median"
        ),
        avg_gross_theoretical_score=(
            "gross_theoretical_score",
            "mean"
        ),
        avg_empirical_tail_prob=(
            "empirical_tail_prob",
            "mean"
        ),
        median_empirical_tail_prob=(
            "empirical_tail_prob",
            "median"
        ),
        avg_lambda_rt=(
            "lambda_rt",
            "mean"
        ),
        avg_net_gap=(
            "net_gap",
            "mean"
        ),
        avg_sigma_hat_s_l=(
            "sigma_hat_s_l",
            "mean"
        ),
        avg_normal_cost_adjusted_threshold=(
            "normal_cost_adjusted_threshold",
            "mean"
        ),
        used_windows=(
            "test_start",
            "nunique"
        ),
        avg_beta=(
            "beta",
            "mean"
        )
    )
    .sort_values(
        [
            "avg_net_theoretical_score",
            "median_net_theoretical_score",
            "used_windows"
        ],
        ascending=[False, False, False]
        )
    .reset_index(drop=True)
    )

    rolling_theory_best = rolling_theory_summary.head(1).copy()

    rolling_theory_summary.to_csv(
        f"{OUTPUT_PREFIX}_project_theoretical_threshold_rolling_summary.csv",
        index=False
    )

    rolling_theory_best.to_csv(
        f"{OUTPUT_PREFIX}_project_theoretical_threshold_rolling_best.csv",
        index=False
    )

    print("\n📌 Rolling theoretical threshold:")
    print(rolling_theory_best[
    [
        "lookback_z",
        "entry_z",
        "exit_z",
        "avg_lambda_rt",
        "avg_net_gap",
        "avg_empirical_tail_prob",
        "avg_net_theoretical_score",
        "avg_normal_cost_adjusted_threshold",
        "used_windows"
    ]
    ].to_string(index=False))
else:
    print("⚠️ Rolling theoretical threshold could not be computed.")




   



# -----------------------------------------
# utility function
# -----------------------------------------
def summarize_sensitivity_result(df_result, mode, lookback_z, entry_z, exit_z, beta_info):
    if df_result is None or df_result.empty:
        return {
            "mode": mode,
            "pair": best_pair_name,
            "lookback_z": lookback_z,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "used_windows": 0,
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "trade_count": np.nan,
            "approx_round_trips": np.nan,
            "active_days": np.nan,
            "avg_hit_rate": np.nan,
            "avg_beta": np.nan,
            "beta_info": beta_info,
            "note": "no valid result"
        }

    metrics = calc_metrics(df_result["strategy_ret_net"])

    trade_count = int(df_result["close_event"].gt(0).sum())
    approx_round_trips = float(trade_count)
    active_days = int(df_result["position_lag"].ne(0).sum())
    avg_hit_rate = df_result["direction_hit"].mean()
    avg_beta = df_result["beta"].mean() if "beta" in df_result.columns else np.nan

    return {
        "mode": mode,
        "pair": best_pair_name,
        "lookback_z": lookback_z,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "used_windows": df_result["test_start"].nunique() if "test_start" in df_result.columns else 1,
        "total_return": metrics["total_return"],
        "annual_return": metrics["annual_return"],
        "annual_vol": metrics["annual_vol"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "trade_count": trade_count,
        "approx_round_trips": approx_round_trips,
        "active_days": active_days,
        "avg_hit_rate": avg_hit_rate,
        "avg_beta": avg_beta,
        "beta_info": beta_info,
        "note": "ok"
    }


def build_static_signal_frame(
    prices,
    s1,
    s2,
    alpha,
    beta,
    lookback_z,
    first_test_start,
    final_test_end,
    test_start_dates
):
    
    pair_data = prices[[s1, s2]].dropna().sort_index().copy()

    full_data = pair_data[
        (pair_data.index >= pd.Timestamp(GLOBAL_START)) &
        (pair_data.index < final_test_end)
    ].copy()

    if full_data.empty:
        return pd.DataFrame()

    spread = compute_spread(
        full_data[s1],
        full_data[s2],
        beta=beta,
        alpha=alpha
    )

    zscore = rolling_zscore(
        spread,
        window=lookback_z
    )

    signal_df = pd.DataFrame(index=full_data.index)

    signal_df["price1"] = full_data[s1]
    signal_df["price2"] = full_data[s2]
    signal_df["spread"] = spread
    signal_df["zscore"] = zscore
    signal_df["alpha"] = alpha
    signal_df["beta"] = beta

    signal_df["train_start"] = pd.Timestamp(
        GLOBAL_START
    ).date()

    signal_df["train_end"] = (
        first_test_start - pd.Timedelta(days=1)
    ).date()

    
    signal_df = signal_df[
        (signal_df.index >= first_test_start) &
        (signal_df.index < final_test_end)
    ].copy()

   
    signal_df["test_start"] = pd.NaT
    signal_df["test_end"] = pd.NaT

    for test_start in test_start_dates:
        test_end = (
            test_start
            + pd.DateOffset(months=TEST_MONTHS)
        )

        mask = (
            (signal_df.index >= test_start) &
            (signal_df.index < test_end)
        )

        signal_df.loc[
            mask,
            "test_start"
        ] = pd.Timestamp(test_start)

        signal_df.loc[
            mask,
            "test_end"
        ] = pd.Timestamp(
            test_end - pd.Timedelta(days=1)
        )

    return signal_df



def build_rolling_signal_frame(
    prices,
    s1,
    s2,
    lookback_z,
    test_start_dates
):

    monthly_frames = []

    for test_start in test_start_dates:

        train_end = test_start

        train_start = (
            train_end
            - pd.DateOffset(months=TRAIN_MONTHS)
        )

        test_end = (
            test_start
            + pd.DateOffset(months=TEST_MONTHS)
        )

        train_data = prices[
            (prices.index >= train_start) &
            (prices.index < train_end)
        ].copy()

        test_data = prices[
            (prices.index >= test_start) &
            (prices.index < test_end)
        ].copy()

        if train_data.empty or test_data.empty:
            continue

        if (
            s1 not in train_data.columns or
            s2 not in train_data.columns
        ):
            continue

        if (
            s1 not in test_data.columns or
            s2 not in test_data.columns
        ):
            continue

        alpha, beta = estimate_alpha_beta_ols(
            train_data,
            s1,
            s2
        )

        if pd.isna(beta):
            continue

        
        warmup_data = (
            train_data[[s1, s2]]
            .dropna()
            .tail(lookback_z + 1)
        )

        live_data = (
            test_data[[s1, s2]]
            .dropna()
            .copy()
        )

        combined_data = pd.concat(
            [warmup_data, live_data],
            axis=0
        )

        combined_data = combined_data[
            ~combined_data.index.duplicated(
                keep="last"
            )
        ].sort_index()

        if len(combined_data) <= lookback_z:
            continue

        spread = compute_spread(
            combined_data[s1],
            combined_data[s2],
            beta=beta,
            alpha=alpha
        )

        zscore = rolling_zscore(
            spread,
            window=lookback_z
        )

      
        live_index = combined_data.index[
            (combined_data.index >= test_start) &
            (combined_data.index < test_end)
        ]

        if len(live_index) == 0:
            continue

        month_df = pd.DataFrame(
            index=live_index
        )

        month_df["price1"] = combined_data.loc[
            live_index,
            s1
        ]

        month_df["price2"] = combined_data.loc[
            live_index,
            s2
        ]

        month_df["spread"] = spread.loc[
            live_index
        ]

        month_df["zscore"] = zscore.loc[
            live_index
        ]

        month_df["alpha"] = alpha
        month_df["beta"] = beta

        month_df["train_start"] = (
            train_start.date()
        )

        month_df["train_end"] = (
            train_end - pd.Timedelta(days=1)
        ).date()

        month_df["test_start"] = pd.Timestamp(test_start)

        month_df["test_end"] = pd.Timestamp(
            test_end - pd.Timedelta(days=1)
        )

        monthly_frames.append(month_df)

    if len(monthly_frames) == 0:
        return pd.DataFrame()

    signal_df = pd.concat(
        monthly_frames,
        axis=0
    ).sort_index()

    signal_df = signal_df[
        ~signal_df.index.duplicated(
            keep="last"
        )
    ]

    return signal_df



def backtest_continuous_signal_path(
    signal_df,
    full_prices,
    s1,
    s2,
    entry_z,
    exit_z,
    cost_per_trade,
    force_close_final=True
):
    if signal_df is None or signal_df.empty:
        return pd.DataFrame()

    df_bt = signal_df.sort_index().copy()

    
    price_history = (
        full_prices[[s1, s2]]
        .dropna()
        .sort_index()
    )

    full_r1 = price_history[s1].pct_change()
    full_r2 = price_history[s2].pct_change()

    df_bt["r1"] = full_r1.reindex(
        df_bt.index
    )

    df_bt["r2"] = full_r2.reindex(
        df_bt.index
    )

    \
    df_bt["position"] = generate_positions(
        df_bt["zscore"],
        entry_z=entry_z,
        exit_z=exit_z
    )

    
    if force_close_final and not df_bt.empty:
        final_index = df_bt.index[-1]
        df_bt.loc[
            final_index,
            "position"
        ] = 0.0

    df_bt["position_lag"] = (
        df_bt["position"]
        .shift(1)
        .fillna(0.0)
    )

    df_bt["spread_ret"] = (
        df_bt["r1"]
        - df_bt["beta"] * df_bt["r2"]
    ) / (
        1.0 + df_bt["beta"].abs()
    )

  
    df_bt["strategy_ret_gross"] = (
        df_bt["position_lag"]
        * df_bt["spread_ret"]
    )

    previous_position = (
        df_bt["position_lag"]
    )

    position_changed = (
        df_bt["position"]
        .ne(previous_position)
    )

    df_bt["open_event"] = (
        df_bt["position"].ne(0)
        & position_changed
    ).astype(int)

    df_bt["close_event"] = (
        previous_position.ne(0)
        & position_changed
    ).astype(int)

    # -----------------------------------------
    # two period turnover
    # -----------------------------------------

    # last period beta
    beta_lag = (
        df_bt["beta"]
        .shift(1)
        .fillna(df_bt["beta"])
    )

    
    # w(q_{t-1}, beta_{t-1})
    previous_scale = (
        1.0 + beta_lag.abs()
    )

    previous_weight_leg1 = (
        previous_position
        / previous_scale
    )

    previous_weight_leg2 = (
        -previous_position
        * beta_lag
        / previous_scale
    )

    
    # w(q_{t-1}, beta_t)
    current_scale = (
        1.0 + df_bt["beta"].abs()
    )

    pre_signal_weight_leg1 = (
        previous_position
        / current_scale
    )

    pre_signal_weight_leg2 = (
        -previous_position
        * df_bt["beta"]
        / current_scale
    )

    
    # w(q_t, beta_t)
    post_signal_weight_leg1 = (
        df_bt["position"]
        / current_scale
    )

    post_signal_weight_leg2 = (
        -df_bt["position"]
        * df_bt["beta"]
        / current_scale
    )

    # updata beta  turnover
    df_bt["beta_rebalance_turnover"] = (
        (
            pre_signal_weight_leg1
            - previous_weight_leg1
        ).abs()
        +
        (
            pre_signal_weight_leg2
            - previous_weight_leg2
        ).abs()
    )

    
    df_bt["signal_turnover"] = (
        (
            post_signal_weight_leg1
            - pre_signal_weight_leg1
        ).abs()
        +
        (
            post_signal_weight_leg2
            - pre_signal_weight_leg2
        ).abs()
    )

    df_bt["turnover"] = (
        df_bt["beta_rebalance_turnover"]
        + df_bt["signal_turnover"]
    )

    df_bt["cost"] = (
        df_bt["turnover"]
        * cost_per_trade
    )

    df_bt["strategy_ret_net"] = (
        df_bt["strategy_ret_gross"]
        - df_bt["cost"]
    )

    df_bt["cum_gross"] = (
        1.0
        + df_bt[
            "strategy_ret_gross"
        ].fillna(0.0)
    ).cumprod()

    df_bt["cum_net"] = (
        1.0
        + df_bt[
            "strategy_ret_net"
        ].fillna(0.0)
    ).cumprod()

    # -----------------------------------------
    # Direction diagnostics
    # -----------------------------------------

    df_bt["predicted_direction"] = (
        df_bt["position_lag"]
    )

    df_bt["actual_direction"] = (
        np.sign(
            df_bt["spread_ret"]
        ).replace(0, np.nan)
    )

    df_bt["direction_hit"] = np.where(
        df_bt["predicted_direction"] == 0,
        np.nan,
        (
            np.sign(
                df_bt["predicted_direction"]
            )
            ==
            np.sign(
                df_bt["spread_ret"]
            )
        ).astype(float)
    )

    df_bt["cum_hit_rate"] = (
        df_bt["direction_hit"]
        .expanding(min_periods=1)
        .mean()
    )

    df_bt["predicted_cum_direction"] = (
        df_bt[
            "predicted_direction"
        ].fillna(0.0).cumsum()
    )

    df_bt["actual_cum_spread_ret"] = (
        df_bt[
            "spread_ret"
        ].fillna(0.0).cumsum()
    )

    return df_bt

# =========================================
# A. static_full_period mode
# =========================================

static_train_start = pd.Timestamp(
    GLOBAL_START
)

static_train_end = first_test_start

static_train_data = prices[
    (prices.index >= static_train_start) &
    (prices.index < static_train_end)
].copy()

static_alpha, static_beta = (
    estimate_alpha_beta_ols(
        static_train_data,
        best_s1,
        best_s2
    )
)

if pd.isna(static_beta):

    print(
        "⚠️ static_full_period 模式无法估计 beta，"
        "将跳过该模式。"
    )

else:

    print(
        f"\nStatic beta estimated from initial "
        f"training window: {static_beta:.4f}"
    )

    for lookback_z_value in LOOKBACK_GRID:

        static_signal_df = (
            build_static_signal_frame(
                prices=prices,
                s1=best_s1,
                s2=best_s2,
                alpha=static_alpha,
                beta=static_beta,
                lookback_z=lookback_z_value,
                first_test_start=first_test_start,
                final_test_end=final_test_end,
                test_start_dates=test_start_dates
            )
        )

        for entry_z_value in ENTRY_Z_GRID:

            for exit_z_value in EXIT_Z_GRID:

                if (
                    exit_z_value
                    >= entry_z_value
                ):

                    sensitivity_summary.append({
                        "mode":
                            "static_full_period",

                        "pair":
                            best_pair_name,

                        "lookback_z":
                            lookback_z_value,

                        "entry_z":
                            entry_z_value,

                        "exit_z":
                            exit_z_value,

                        "used_windows":
                            0,

                        "total_return":
                            np.nan,

                        "annual_return":
                            np.nan,

                        "annual_vol":
                            np.nan,

                        "sharpe":
                            np.nan,

                        "max_drawdown":
                            np.nan,

                        "trade_count":
                            np.nan,

                        "approx_round_trips":
                            np.nan,

                        "active_days":
                            np.nan,

                        "avg_hit_rate":
                            np.nan,

                        "avg_beta":
                            static_beta,

                        "beta_info": (
                            "static beta from first "
                            "12-month training window; "
                            "continuous OOS position path"
                        ),

                        "note": (
                            "skipped because "
                            "exit_z >= entry_z"
                        )
                    })

                    continue

                result_test = (
                    backtest_continuous_signal_path(
                        signal_df=
                            static_signal_df,

                        full_prices=
                            prices,

                        s1=
                            best_s1,

                        s2=
                            best_s2,

                        entry_z=
                            entry_z_value,

                        exit_z=
                            exit_z_value,

                        cost_per_trade=
                            COST_PER_TRADE,

                        force_close_final=
                            True
                    )
                )

                if result_test.empty:

                    summary_row = (
                        summarize_sensitivity_result(
                            None,

                            mode=
                                "static_full_period",

                            lookback_z=
                                lookback_z_value,

                            entry_z=
                                entry_z_value,

                            exit_z=
                                exit_z_value,

                            beta_info=(
                                "static beta from first "
                                "12-month training window; "
                                "continuous OOS position path"
                            )
                        )
                    )

                else:

                    result_test["mode"] = (
                        "static_full_period"
                    )

                    result_test["pair"] = (
                        best_pair_name
                    )

                    result_test["s1"] = best_s1
                    result_test["s2"] = best_s2

                    result_test["lookback_z"] = (
                        lookback_z_value
                    )

                    result_test["entry_z"] = (
                        entry_z_value
                    )

                    result_test["exit_z"] = (
                        exit_z_value
                    )

                    summary_row = (
                        summarize_sensitivity_result(
                            result_test,

                            mode=
                                "static_full_period",

                            lookback_z=
                                lookback_z_value,

                            entry_z=
                                entry_z_value,

                            exit_z=
                                exit_z_value,

                            beta_info=(
                                "static beta from first "
                                "12-month training window; "
                                "continuous OOS position path"
                            )
                        )
                    )

                    sensitivity_daily_results.append(
                        result_test
                    )

                sensitivity_summary.append(
                    summary_row
                )






# =========================================
# B. rolling_calibrated mode
# 固定 best pair
# 每个月用过去 12 个月重新估计 alpha 和 beta
# 持仓跨月份连续传递
# =========================================

for lookback_z_value in LOOKBACK_GRID:

    rolling_signal_df = (
        build_rolling_signal_frame(
            prices=prices,
            s1=best_s1,
            s2=best_s2,
            lookback_z=lookback_z_value,
            test_start_dates=test_start_dates
        )
    )

    for entry_z_value in ENTRY_Z_GRID:

        for exit_z_value in EXIT_Z_GRID:

            if (
                exit_z_value
                >= entry_z_value
            ):

                sensitivity_summary.append({
                    "mode":
                        "rolling_calibrated",

                    "pair":
                        best_pair_name,

                    "lookback_z":
                        lookback_z_value,

                    "entry_z":
                        entry_z_value,

                    "exit_z":
                        exit_z_value,

                    "used_windows":
                        0,

                    "total_return":
                        np.nan,

                    "annual_return":
                        np.nan,

                    "annual_vol":
                        np.nan,

                    "sharpe":
                        np.nan,

                    "max_drawdown":
                        np.nan,

                    "trade_count":
                        np.nan,

                    "approx_round_trips":
                        np.nan,

                    "active_days":
                        np.nan,

                    "avg_hit_rate":
                        np.nan,

                    "avg_beta":
                        np.nan,

                    "beta_info": (
                        "rolling beta re-estimated every "
                        "12-month training window; "
                        "continuous OOS position path"
                    ),

                    "note": (
                        "skipped because "
                        "exit_z >= entry_z"
                    )
                })

                continue

            result_test = (
                backtest_continuous_signal_path(
                    signal_df=
                        rolling_signal_df,

                    full_prices=
                        prices,

                    s1=
                        best_s1,

                    s2=
                        best_s2,

                    entry_z=
                        entry_z_value,

                    exit_z=
                        exit_z_value,

                    cost_per_trade=
                        COST_PER_TRADE,

                    force_close_final=
                        True
                )
            )

            if result_test.empty:

                summary_row = (
                    summarize_sensitivity_result(
                        None,

                        mode=
                            "rolling_calibrated",

                        lookback_z=
                            lookback_z_value,

                        entry_z=
                            entry_z_value,

                        exit_z=
                            exit_z_value,

                        beta_info=(
                            "rolling beta re-estimated "
                            "every 12-month training window; "
                            "continuous OOS position path"
                        )
                    )
                )

            else:

                result_test["mode"] = (
                    "rolling_calibrated"
                )

                result_test["pair"] = (
                    best_pair_name
                )

                result_test["s1"] = best_s1
                result_test["s2"] = best_s2

                result_test["lookback_z"] = (
                    lookback_z_value
                )

                result_test["entry_z"] = (
                    entry_z_value
                )

                result_test["exit_z"] = (
                    exit_z_value
                )

                summary_row = (
                    summarize_sensitivity_result(
                        result_test,

                        mode=
                            "rolling_calibrated",

                        lookback_z=
                            lookback_z_value,

                        entry_z=
                            entry_z_value,

                        exit_z=
                            exit_z_value,

                        beta_info=(
                            "rolling beta re-estimated "
                            "every 12-month training window; "
                            "continuous OOS position path"
                        )
                    )
                )

                sensitivity_daily_results.append(
                    result_test
                )

            sensitivity_summary.append(
                summary_row
            )


# =========================================
# C. stored full CSV
# =========================================

sensitivity_summary_df = pd.DataFrame(
    sensitivity_summary
)

sensitivity_summary_df = (
    sensitivity_summary_df
    .sort_values(
        [
            "mode",
            "lookback_z",
            "entry_z",
            "exit_z"
        ]
    )
    .reset_index(drop=True)
)

sensitivity_summary_df.to_csv(
    (
        f"{OUTPUT_PREFIX}_best_pair_zscore_"
        f"static_vs_rolling_full_results.csv"
    ),
    index=False
)

print("\n✅ stored the full  CSV:")
print(
    f"{OUTPUT_PREFIX}_best_pair_zscore_"
    f"static_vs_rolling_full_results.csv"
)

# -----------------------------------------
# stored valid daily-level results
# -----------------------------------------

if len(sensitivity_daily_results) > 0:

    sensitivity_daily_df = (
        pd.concat(
            sensitivity_daily_results,
            axis=0
        )
        .sort_index()
    )

    sensitivity_daily_df.to_csv(
        (
            f"{OUTPUT_PREFIX}_best_pair_zscore_"
            f"static_vs_rolling_daily_results.csv"
        ),
        index=True
    )

    print("✅ saved daily-level CSV:")
    print(
        f"{OUTPUT_PREFIX}_best_pair_zscore_"
        f"static_vs_rolling_daily_results.csv"
    )

    # -----------------------------------------
    # net statistic
    # -----------------------------------------

    strat_ret_check = (
        sensitivity_daily_df[
            "strategy_ret_net"
        ]
        .dropna()
    )

    print(
        "\n📌 sensitivity analysis："
        "Description statistics of daily net yield of the strategy："
    )

    print(
        strat_ret_check.describe()
    )

    # -----------------------------------------
    # extreme daily return
    # -----------------------------------------

    extreme_mask = (
        sensitivity_daily_df[
            "strategy_ret_net"
        ].abs()
        > 0.3
    )

    extreme_df = (
        sensitivity_daily_df[
            extreme_mask
        ]
        .copy()
    )

    extreme_df = extreme_df.sort_values(
        "strategy_ret_net",
        key=abs,
        ascending=False
    )

    print(
        "\n⚠️ The number of days when the net profit of the single-day strategy exceeded ±30%："
        f"{len(extreme_df)}"
    )

    if len(extreme_df) > 0:

        cols_to_show = [
            c
            for c in [
                "pair",
                "mode",
                "lookback_z",
                "entry_z",
                "exit_z",
                "spread_ret",
                "strategy_ret_gross",
                "strategy_ret_net",
                "cost",
                "turnover",
                "beta_rebalance_turnover",
                "signal_turnover",
                "position_lag",
                "position",
                "beta"
            ]
            if c in extreme_df.columns
        ]

        print(
            extreme_df[
                cols_to_show
            ]
            .head(30)
            .to_string()
        )

    extreme_df.to_csv(
        (
            f"{OUTPUT_PREFIX}_"
            f"extreme_strategy_returns.csv"
        ),
        index=True
    )

else:

    print(
        "⚠️ sensitivity_daily_results 为空，"
        "未生成 daily-level sensitivity CSV。"
    )






# =========================================
# D. Select the optimal parameter combination for each mode
# =========================================

valid_sensitivity_df = sensitivity_summary_df[
    (sensitivity_summary_df["note"] == "ok") &
    (sensitivity_summary_df["sharpe"].notna())
].copy()

if not valid_sensitivity_df.empty:
    best_params_by_mode = (
        valid_sensitivity_df
        .sort_values(
            ["mode", "sharpe", "total_return", "max_drawdown", "trade_count"],
            ascending=[True, False, False, False, False]
        )
        .groupby("mode")
        .head(1)
        .reset_index(drop=True)
    )

    best_params_by_mode.to_csv(
        f"{OUTPUT_PREFIX}_best_pair_zscore_best_params_by_mode.csv",
        index=False
    )
    print("\n🏆 the optimal parameter combination for each mode：")
    print(best_params_by_mode[
        ["mode", "lookback_z", "entry_z", "exit_z",
         "sharpe", "total_return", "max_drawdown", "trade_count"]
    ].to_string(index=False))

    print(f"✅ save：{OUTPUT_PREFIX}_best_pair_zscore_best_params_by_mode.csv")
else:
    print("⚠️ There is no valid parameter combination that can be sorted.")


# =========================================
# D2. Compare theoretical threshold with empirical sensitivity optimum
# =========================================

comparison_rows = []

if "static_theory_best" in globals() and not static_theory_best.empty:
    comparison_rows.append({
        "type": "static_theoretical",
        "lookback_z": static_theory_best.iloc[0]["lookback_z"],
        "entry_z": static_theory_best.iloc[0]["entry_z"],
        "exit_z": static_theory_best.iloc[0]["exit_z"],
        "criterion": ("max cost-adjusted project-specific "
                      "theoretical score"
                      ),
        "score": static_theory_best.iloc[0][
            "net_theoretical_score"
            ],
        "lambda_rt": static_theory_best.iloc[0]["lambda_rt"],
        "net_gap": static_theory_best.iloc[0]["net_gap"]
    })

if "rolling_theory_best" in globals() and not rolling_theory_best.empty:
    comparison_rows.append({
        "type": "rolling_theoretical",
        "lookback_z": rolling_theory_best.iloc[0]["lookback_z"],
        "entry_z": rolling_theory_best.iloc[0]["entry_z"],
        "exit_z": rolling_theory_best.iloc[0]["exit_z"],
        "criterion": (
            "max average cost-adjusted "
            "project-specific theoretical score"
            ),
        "score": rolling_theory_best.iloc[0][
            "avg_net_theoretical_score"
        ],
        "lambda_rt": rolling_theory_best.iloc[0][
        "avg_lambda_rt"
        ],
        "net_gap": rolling_theory_best.iloc[0][
            "avg_net_gap"
            ]
    })

if "best_params_by_mode" in globals() and not best_params_by_mode.empty:
    for _, row in best_params_by_mode.iterrows():
        comparison_rows.append({
            "type": row["mode"] + "_empirical",
            "lookback_z": row["lookback_z"],
            "entry_z": row["entry_z"],
            "exit_z": row["exit_z"],
            "criterion": "max out-of-sample Sharpe ratio",
            "score": row["sharpe"]
        })

threshold_comparison_df = pd.DataFrame(comparison_rows)

threshold_comparison_df.to_csv(
    f"{OUTPUT_PREFIX}_theoretical_vs_empirical_threshold_comparison.csv",
    index=False
)

print("\n📊 Theoretical vs empirical threshold comparison:")
print(threshold_comparison_df.to_string(index=False))



# =========================================
# E. Sharpe Heatmap
# Since there are currently two modes, 8 charts will be generated:
# 4 lookbacks × 2 modes
# Horizontal axis: exit_z, vertical axis: entry_z, color: Sharpe
# Green = good, red = bad
# =========================================

for mode_name in ["static_full_period", "rolling_calibrated"]:
    for lookback_z_value in LOOKBACK_GRID:

        heatmap_df = sensitivity_summary_df[
            (sensitivity_summary_df["mode"] == mode_name) &
            (sensitivity_summary_df["lookback_z"] == lookback_z_value)
        ].copy()

        pivot_df = heatmap_df.pivot(
            index="entry_z",
            columns="exit_z",
            values="sharpe"
        )

        pivot_df = pivot_df.sort_index().sort_index(axis=1)

        plt.figure(figsize=(8, 6))

        heat_values = np.ma.masked_invalid(pivot_df.values)

        im = plt.imshow(
            heat_values,
            cmap="RdYlGn",
            aspect="auto",
            origin="lower"
        )

        plt.colorbar(im, label="Sharpe ratio")

        plt.xticks(
            ticks=np.arange(len(pivot_df.columns)),
            labels=[str(x) for x in pivot_df.columns]
        )

        plt.yticks(
            ticks=np.arange(len(pivot_df.index)),
            labels=[str(x) for x in pivot_df.index]
        )

        plt.xlabel("Exit Z-score threshold")
        plt.ylabel("Entry Z-score threshold")
        plt.title(
            f"Sharpe Heatmap: {best_pair_name}\n"
            f"Mode = {mode_name}, Lookback = {lookback_z_value}"
        )

        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                val = pivot_df.iloc[i, j]
                if pd.notna(val):
                    plt.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8
                    )

        plt.tight_layout()

        heatmap_filename = (
            f"{OUTPUT_PREFIX}_best_pair_zscore_heatmap_"
            f"{mode_name}_lookback_{lookback_z_value}.png"
        )

        plt.savefig(heatmap_filename, dpi=150)
        plt.show()

        print(f"✅ saved heatmap: {heatmap_filename}")


# =========================================
# F. Additional Figure: Comparison of optimal Sharpe ratios between static and rolling methods under the same lookback period
# =========================================

if not valid_sensitivity_df.empty:
    best_by_mode_lookback = (
        valid_sensitivity_df
        .sort_values(
            ["mode", "lookback_z", "sharpe"],
            ascending=[True, True, False]
        )
        .groupby(["mode", "lookback_z"])
        .head(1)
        .reset_index(drop=True)
    )

    best_by_mode_lookback.to_csv(
        f"{OUTPUT_PREFIX}_best_pair_zscore_best_by_mode_and_lookback.csv",
        index=False
    )

    plt.figure(figsize=(8, 4))

    for mode_name, group_df in best_by_mode_lookback.groupby("mode"):
        group_df = group_df.sort_values("lookback_z")
        plt.plot(
            group_df["lookback_z"],
            group_df["sharpe"],
            marker="o",
            label=mode_name
        )

    plt.title(f"Best Sharpe by Lookback: Static vs Rolling\n{best_pair_name}")
    plt.xlabel("Lookback window")
    plt.ylabel("Best Sharpe ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    compare_filename = f"{OUTPUT_PREFIX}_best_pair_zscore_static_vs_rolling_best_sharpe.png"

    plt.savefig(compare_filename, dpi=150)
    plt.show()

    print(f"✅ Saved：{OUTPUT_PREFIX}_best_pair_zscore_best_by_mode_and_lookback.csv")
    print(f"✅ Saved：{compare_filename}")

print("\n✅ static full-period + rolling calibrated z-score sensitivity 全部完成。")
    

    




# =========================================
# 10. Combined level of returns (merging the equal-weighted top 5 selections for the current month)
# =========================================
portfolio_daily = (
    all_test_results_df
    .groupby(all_test_results_df.index)["strategy_ret_net"]
    .mean()
    .to_frame("portfolio_ret_net")
    .sort_index()
)

portfolio_daily["portfolio_cum_net"] = (1 + portfolio_daily["portfolio_ret_net"].fillna(0)).cumprod()
portfolio_daily.to_csv(f"{OUTPUT_PREFIX}_portfolio_daily_returns.csv", index=True)

portfolio_metrics = calc_metrics(portfolio_daily["portfolio_ret_net"])
portfolio_metrics_df = pd.DataFrame([portfolio_metrics])
portfolio_metrics_df.to_csv(f"{OUTPUT_PREFIX}_portfolio_metrics.csv", index=False)

# =========================================
# 11. Drawing: Only draw for the "top 5 pairs with the highest occurrence frequency"
# =========================================
for pair_name in top5_pairs_list:
    pair_df = top5_test_df[top5_test_df["pair"] == pair_name].copy()
    if pair_df.empty:
        continue

    pair_df = pair_df.sort_index().copy()
    pair_df["cum_gross_all"] = (1 + pair_df["strategy_ret_gross"].fillna(0)).cumprod()
    pair_df["cum_net_all"] = (1 + pair_df["strategy_ret_net"].fillna(0)).cumprod()

    safe_name = pair_name.replace(" vs ", "_vs_").replace(".", "_")

    # figure1：Spread
    plt.figure(figsize=(12, 4))
    plt.plot(pair_df.index, pair_df["spread"], label="Spread")
    plt.axhline(pair_df["spread"].mean(), linestyle="--", label="Mean")
    plt.title(f"Out-of-Sample Spread: {pair_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{safe_name}_spread.png", dpi=150)
    plt.show()

    # figure2：Z-score
    plt.figure(figsize=(12, 4))
    plt.plot(pair_df.index, pair_df["zscore"], label="Z-score")
    plt.axhline(0, linewidth=1)
    plt.axhline(ENTRY_Z, linestyle="--", label=f"+{ENTRY_Z}")
    plt.axhline(-ENTRY_Z, linestyle="--", label=f"-{ENTRY_Z}")
    plt.axhline(EXIT_Z, linestyle=":", label=f"exit +{EXIT_Z}")
    plt.axhline(-EXIT_Z, linestyle=":", label=f"exit -{EXIT_Z}")
    plt.title(f"Out-of-Sample Z-score: {pair_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{safe_name}_zscore.png", dpi=150)
    plt.show()

    # figure3：Position
    plt.figure(figsize=(12, 3))
    plt.plot(pair_df.index, pair_df["position"], label="Position")
    plt.axhline(1, linestyle="--")
    plt.axhline(0, linewidth=1)
    plt.axhline(-1, linestyle="--")
    plt.title(f"Out-of-Sample Position: {pair_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{safe_name}_position.png", dpi=150)
    plt.show()

    # figure4：Cumulative Return
    plt.figure(figsize=(12, 4))
    plt.plot(pair_df.index, pair_df["cum_gross_all"], label="Gross cumulative return")
    plt.plot(pair_df.index, pair_df["cum_net_all"], label="Net cumulative return")
    plt.title(f"Out-of-Sample Cumulative Return: {pair_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{safe_name}_cum_return.png", dpi=150)
    plt.show()

    # figure5：Predicted direction vs. Actual result
    # This is a directional prediction, not a point prediction model.
    plt.figure(figsize=(12, 4))
    plt.plot(pair_df.index, pair_df["predicted_cum_direction"], label="Predicted cumulative direction")
    plt.plot(pair_df.index, pair_df["actual_cum_spread_ret"], label="Actual cumulative spread return")
    plt.title(f"Predicted vs Actual (direction-based): {pair_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{safe_name}_pred_vs_actual.png", dpi=150)
    plt.show()

    # figure6：Cumulative hit rate
    plt.figure(figsize=(12, 4))
    plt.plot(pair_df.index, pair_df["cum_hit_rate"], label="Cumulative hit rate")
    plt.axhline(0.5, linestyle="--", label="0.5 benchmark")
    plt.title(f"Signal Hit Rate: {pair_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{safe_name}_hit_rate.png", dpi=150)
    plt.show()

# figure7：The top 5 frequency bar chart
plt.figure(figsize=(10, 5))
plt.bar(top5_most_frequent_pairs["pair"], top5_most_frequent_pairs["occurrence_count"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 5 Most Frequently Selected Pairs")
plt.ylabel("Occurrence Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_top5_occurrence_bar.png", dpi=150)
plt.show()

# figure8：portfolio_cum_return
plt.figure(figsize=(12, 4))
plt.plot(portfolio_daily.index, portfolio_daily["portfolio_cum_net"], label="Portfolio cumulative net return")
plt.title("Dynamic Top-5 Portfolio Out-of-Sample Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_portfolio_cum_return.png", dpi=150)
plt.show()

print("\n✅ All completed.")
print("Main output file：")
print(f"1) {OUTPUT_PREFIX}_monthly_selected_top5_pairs.csv")
print(f"2) {OUTPUT_PREFIX}_top5_most_frequent_pairs.csv")
print(f"3) {OUTPUT_PREFIX}_all_test_results_daily.csv")
print(f"4) {OUTPUT_PREFIX}_top5_pairs_annual_returns.csv")
print(f"5) {OUTPUT_PREFIX}_top5_pairs_total_metrics.csv")
print(f"6) {OUTPUT_PREFIX}_portfolio_daily_returns.csv")
print(f"7) {OUTPUT_PREFIX}_portfolio_metrics.csv")