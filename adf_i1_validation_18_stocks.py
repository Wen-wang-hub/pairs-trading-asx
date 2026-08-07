from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

GLOBAL_START = "2022-01-01"
GLOBAL_END = "2026-01-01"  # exclusive in yfinance

SIGNIFICANCE_LEVEL = 0.05
MIN_OBSERVATIONS = 100

# Level log prices often contain deterministic drift, so use a constant
# and linear trend. First differences use a constant only.
LEVEL_REGRESSION = "ct"
DIFF_REGRESSION = "c"

OUTPUT_DIR = Path("adf_i1_validation_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAS_TICKERS = [
    "WDS.AX", "STO.AX", "BPT.AX", "AEL.AX", "STX.AX", "TBN.AX",
    "BTL.AX", "CTP.AX", "COI.AX", "CUE.AX", "TDO.AX", "IPB.AX",
    "APA.AX", "AGL.AX", "ORG.AX", "ALD.AX", "VEA.AX", "EWC.AX",
]

TICKER_HISTORY_MAP = {
    "AEL.AX": ["AEL.AX", "COE.AX"],
    "BTL.AX": ["BTL.AX", "EEG.AX"],
    "IPB.AX": ["IPB.AX", "FEL.AX"],
}

def build_download_list() -> list[str]:
    tickers: list[str] = []
    for canonical_ticker in GAS_TICKERS:
        tickers.extend(TICKER_HISTORY_MAP.get(canonical_ticker, [canonical_ticker]))
    return list(dict.fromkeys(tickers))


def extract_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError(
            "The downloaded dataset is empty. Check the ticker symbols or network connection."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("The yfinance response does not contain Close prices.")
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw.copy()

    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def construct_canonical_price_panel(
    close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.DataFrame(index=close.index)
    availability_records: list[dict[str, object]] = []

    for canonical_ticker in GAS_TICKERS:
        candidate_tickers = TICKER_HISTORY_MAP.get(
            canonical_ticker,
            [canonical_ticker],
        )

        combined_series: Optional[pd.Series] = None
        used_sources: list[str] = []

        for source_ticker in candidate_tickers:
            if source_ticker not in close.columns:
                continue

            source_series = close[source_ticker].copy()
            if source_series.dropna().empty:
                continue

            if combined_series is None:
                combined_series = source_series
            else:
                combined_series = combined_series.combine_first(source_series)

            used_sources.append(source_ticker)

        if combined_series is None or combined_series.dropna().empty:
            availability_records.append({
                "ticker": canonical_ticker,
                "used_sources": "",
                "available": False,
                "first_available_date": pd.NaT,
                "last_available_date": pd.NaT,
                "n_observations": 0,
            })
            continue

        prices[canonical_ticker] = combined_series
        valid_series = combined_series.dropna()

        availability_records.append({
            "ticker": canonical_ticker,
            "used_sources": ", ".join(used_sources),
            "available": True,
            "first_available_date": valid_series.index.min().date(),
            "last_available_date": valid_series.index.max().date(),
            "n_observations": int(len(valid_series)),
        })

    prices = (
        prices
        .dropna(axis=1, how="all")
        .sort_index()
        .reindex(sorted(prices.columns), axis=1)
    )

    return prices, pd.DataFrame(availability_records)


# ==========================================================
#  ADF utilities
# ==========================================================

def run_adf(series: pd.Series, regression: str) -> dict[str, float | int]:
    clean_series = (
        series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    )

    if len(clean_series) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(clean_series)} usable observations; "
            f"at least {MIN_OBSERVATIONS} are required."
        )

    if clean_series.nunique() <= 1:
        raise ValueError("The series is constant and cannot be tested.")

    result = adfuller(
        clean_series,
        regression=regression,
        autolag="AIC",
    )

    critical_values = result[4]

    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "n_observations": int(result[3]),
        "critical_value_1pct": float(critical_values["1%"]),
        "critical_value_5pct": float(critical_values["5%"]),
        "critical_value_10pct": float(critical_values["10%"]),
    }


def test_ticker_i1(ticker: str, price_series: pd.Series) -> dict[str, object]:
    valid_prices = (
        price_series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    )
    valid_prices = valid_prices[valid_prices > 0]

    base_result: dict[str, object] = {
        "ticker": ticker,
        "raw_price_observations": int(len(valid_prices)),
        "first_price_date": (
            valid_prices.index.min().date() if not valid_prices.empty else pd.NaT
        ),
        "last_price_date": (
            valid_prices.index.max().date() if not valid_prices.empty else pd.NaT
        ),
    }

    if len(valid_prices) < MIN_OBSERVATIONS:
        return {
            **base_result,
            "status": "insufficient observations",
            "i1_confirmed": False,
        }

    log_price = np.log(valid_prices)
    first_difference = log_price.diff().dropna()

    try:
        level_result = run_adf(log_price, regression=LEVEL_REGRESSION)
        diff_result = run_adf(first_difference, regression=DIFF_REGRESSION)

        level_has_unit_root = level_result["p_value"] >= SIGNIFICANCE_LEVEL
        first_difference_stationary = diff_result["p_value"] < SIGNIFICANCE_LEVEL
        i1_confirmed = level_has_unit_root and first_difference_stationary

        return {
            **base_result,
            "status": "ok",
            "level_regression": LEVEL_REGRESSION,
            "level_adf_statistic": level_result["adf_statistic"],
            "level_p_value": level_result["p_value"],
            "level_used_lag": level_result["used_lag"],
            "level_adf_observations": level_result["n_observations"],
            "level_critical_value_5pct": level_result["critical_value_5pct"],
            "level_reject_unit_root": not level_has_unit_root,
            "diff_regression": DIFF_REGRESSION,
            "diff_adf_statistic": diff_result["adf_statistic"],
            "diff_p_value": diff_result["p_value"],
            "diff_used_lag": diff_result["used_lag"],
            "diff_adf_observations": diff_result["n_observations"],
            "diff_critical_value_5pct": diff_result["critical_value_5pct"],
            "diff_reject_unit_root": first_difference_stationary,
            "i1_confirmed": i1_confirmed,
        }

    except Exception as exc:
        return {
            **base_result,
            "status": f"ADF test failed: {exc}",
            "i1_confirmed": False,
        }


# ==========================================================
# 4. Main workflow
# ==========================================================

def main() -> None:
    download_tickers = build_download_list()

    print("=" * 72)
    print("ADF I(1) VALIDATION FOR THE 18-STOCK ASX ENERGY UNIVERSE")
    print("=" * 72)
    print(f"Canonical universe size: {len(GAS_TICKERS)}")
    print(f"Sample: {GLOBAL_START} to {GLOBAL_END} (exclusive end)")
    print(f"Significance level: {SIGNIFICANCE_LEVEL:.0%}")

    raw = yf.download(
        download_tickers,
        start=GLOBAL_START,
        end=GLOBAL_END,
        auto_adjust=True,
        progress=False,
    )

    close = extract_close_prices(raw)
    prices, availability_df = construct_canonical_price_panel(close)

    availability_path = OUTPUT_DIR / "ticker_data_availability_18_stocks.csv"
    prices_path = OUTPUT_DIR / "merged_prices_18_stocks.csv"

    availability_df.to_csv(availability_path, index=False)
    prices.to_csv(prices_path, index=True)

    missing_tickers = [ticker for ticker in GAS_TICKERS if ticker not in prices.columns]
    if missing_tickers:
        print("Warning: no usable price series for: " + ", ".join(missing_tickers))

    results = [
        test_ticker_i1(ticker=ticker, price_series=prices[ticker])
        for ticker in prices.columns
    ]

    results_df = pd.DataFrame(results)

    preferred_columns = [
        "ticker", "status", "raw_price_observations",
        "first_price_date", "last_price_date",
        "level_regression", "level_adf_statistic", "level_p_value",
        "level_used_lag", "level_adf_observations",
        "level_critical_value_5pct", "level_reject_unit_root",
        "diff_regression", "diff_adf_statistic", "diff_p_value",
        "diff_used_lag", "diff_adf_observations",
        "diff_critical_value_5pct", "diff_reject_unit_root",
        "i1_confirmed",
    ]

    for column in preferred_columns:
        if column not in results_df.columns:
            results_df[column] = np.nan

    results_df = (
        results_df[preferred_columns]
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    results_path = OUTPUT_DIR / "adf_i1_results_18_stocks.csv"
    results_df.to_csv(results_path, index=False)

    valid_results = results_df[results_df["status"] == "ok"].copy()
    n_tested = len(valid_results)
    n_i1 = int(valid_results["i1_confirmed"].fillna(False).sum())
    n_not_i1 = n_tested - n_i1

    non_i1_tickers = valid_results.loc[
        ~valid_results["i1_confirmed"].fillna(False),
        "ticker",
    ].tolist()

    failed_tickers = results_df.loc[
        results_df["status"] != "ok",
        ["ticker", "status"],
    ]

    summary_lines = [
        "ADF I(1) VALIDATION SUMMARY",
        "=" * 40,
        f"Canonical stock universe: {len(GAS_TICKERS)}",
        f"Successfully tested stocks: {n_tested}",
        f"I(1) confirmed: {n_i1}",
        f"I(1) not confirmed: {n_not_i1}",
        f"Significance level: {SIGNIFICANCE_LEVEL:.0%}",
        f"Level specification: regression='{LEVEL_REGRESSION}'",
        f"First-difference specification: regression='{DIFF_REGRESSION}'",
        "",
        "Classification rule:",
        (
            "I(1) confirmed when the level ADF p-value is at least "
            f"{SIGNIFICANCE_LEVEL:.2f} and the first-difference ADF "
            f"p-value is below {SIGNIFICANCE_LEVEL:.2f}."
        ),
        "",
        "Stocks for which I(1) was not confirmed:",
    ]

    if non_i1_tickers:
        summary_lines.extend(f"- {ticker}" for ticker in non_i1_tickers)
    else:
        summary_lines.append("- None")

    summary_lines.extend(["", "Stocks not successfully tested:"])

    if failed_tickers.empty:
        summary_lines.append("- None")
    else:
        for _, row in failed_tickers.iterrows():
            summary_lines.append(f"- {row['ticker']}: {row['status']}")

    summary_path = OUTPUT_DIR / "adf_i1_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print(results_df[
        ["ticker", "level_p_value", "diff_p_value", "i1_confirmed", "status"]
    ].to_string(index=False))

    print()
    print("=" * 72)
    print(f"I(1) confirmed for {n_i1} of {n_tested} tested stocks.")
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {prices_path}")
    print(f"Saved: {availability_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
