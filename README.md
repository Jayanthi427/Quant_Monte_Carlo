
Monte Carlo Quantitative Risk &amp; Pricing Engine with Black-Scholes Benchmarking &amp; VaR Backtesting
# Quantitative Risk & Pricing Engine

A verified Python quantitative finance library for derivative pricing via Monte Carlo simulation, analytical Black-Scholes benchmarking, and VaR model backtesting using statistical coverage tests.

## Key Features

* **Option Pricing Engine**: European Call/Put pricing via Black-Scholes analytical formulas and Monte Carlo simulation with empirical confidence intervals.
* **Risk Backtesting Framework**: Statistical evaluation of Value-at-Risk (VaR) models using Kupiec's Proportion of Failures (POF) test and Christoffersen's Independence test.
* **Test Suite**: Fully verified unit test suite ensuring mathematical convergence and statistical accuracy.

## Project Structure

```text
Quant_Monte_Carlo/
│
├── quant_engine/
│   ├── pricing.py        # Black-Scholes & Monte Carlo option pricing
│   └── backtesting.py    # Kupiec POF & Christoffersen tests
│
└── tests/
    └── test_engine.py    # Pytest verification suite
