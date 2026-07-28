# Pairs Trading Strategy in Financial Market

This repository contains the Python implementation supporting the Master's thesis

**Pairs Trading Strategy in Financial Market**

School of Mathematics and Statistics  
The University of New South Wales (UNSW Sydney)

Author:
Wen Wang

Supervisor:
Dr Ruyi Liu

---

## Project Overview

This project develops a cointegration-based pairs trading framework for ASX-listed energy stocks.

The methodology includes

- Rolling train-test framework
- OLS hedge-ratio estimation
- Engle–Granger cointegration test
- ADF residual stationarity test
- Z-score trading strategy
- Transaction-cost-adjusted backtesting
- Random-seed robustness analysis
- Threshold sensitivity analysis

The best-performing trading pair is

VEN.AX – COI.AX

under the transaction-cost-adjusted Sharpe ratio.

---

## Repository Structure

code/
Python source code

results/
Generated CSV files

figures/
Figures used in the thesis

---

## Data

Price data are downloaded directly from Yahoo Finance using the yfinance package.

The data are therefore not redistributed in this repository.

---

## Requirements

Python 3.11+

Main packages

- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- seaborn
- yfinance

---

## Citation

If you use this repository, please cite the associated Master's thesis.
