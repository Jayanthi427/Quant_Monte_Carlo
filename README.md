# Quantitative Risk & Pricing Engine

An end-to-end Python quantitative finance framework built for option derivative pricing, stochastic simulation, Value-at-Risk (VaR) backtesting, and model diagnostics.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/tests-passing-brightgreen.svg)

---

## Technical Overview

This library provides a modular environment designed for pricing financial derivatives and backtesting risk metrics against empirical market benchmarks. Key architectural highlights include:

* **Analytical & Stochastic Pricing**: Analytical European option benchmarking via Black-Scholes formulas alongside Monte Carlo simulation engines equipped with confidence intervals and convergence diagnostics.
* **Risk Backtesting Framework**: Formal statistical evaluation of Value-at-Risk (VaR) models using Kupiec's Proportion of Failures (POF) coverage test and Christoffersen's Independence test.
* **Interactive Dashboard & Visualization**: Streamlit-based interactive application interface paired with custom plotting modules for stochastic path generation, yield curves, and payoff profiles.
* **Automated Verification**: Pytest-backed verification suite confirming mathematical convergence and pipeline data integrity.

---

## Directory Architecture

```text
Quant_Monte_Carlo/
│
├── quant_engine/
│   ├── __init__.py
│   ├── pricing.py         # Black-Scholes analytical formulas & Monte Carlo engine
│   ├── backtesting.py     # Kupiec POF & Christoffersen VaR statistical tests
│   ├── risk_engine.py     # Value-at-Risk (VaR) & Expected Shortfall (ES) analytics
│   ├── simulation.py      # Stochastic process generators (GBM, jump-diffusion)
│   ├── models.py          # Asset pricing structures & volatility models
│   ├── diagnostics.py     # Model stability, convergence & error metrics
│   ├── data.py            # Market data pipeline & input processing
│   ├── visualisation.py   # Plotting utilities for paths, returns, and payoffs
│   └── app.py             # Interactive dashboard application
│
├── tests/
│   ├── __init__.py
│   ├── test_engine.py     # Mathematical precision & model unit tests
│   └── test_integrity.py  # Data pipeline & integration verification tests
│
├── .gitignore
└── README.md
