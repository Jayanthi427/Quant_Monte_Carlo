Quant_Monte_Carlo/
│
├── quant_engine/
│   ├── pricing.py        # Black-Scholes & Monte Carlo option pricing
│   ├── backtesting.py    # Kupiec POF & Christoffersen tests
│   ├── risk_engine.py    # Value-at-Risk (VaR) & Expected Shortfall (ES)
│   ├── simulation.py    # Path generation & stochastic process engine
│   ├── visualisation.py # Plotting & yield/payoff charts
│   ├── models.py         # Underlying asset & volatility models
│   ├── diagnostics.py    # Model convergence & accuracy checks
│   ├── data.py           # Data fetching & market inputs
│   └── app.py            # Interactive Dashboard / Application interface
│
└── tests/
    ├── test_engine.py    # Core test suite
    └── test_integrity.py # Integration & pipeline integrity tests
