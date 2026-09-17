# Quant Monte Carlo

### Stochastic Simulation, Derivatives Pricing, and Quantitative Risk Analysis

Quant Monte Carlo is a quantitative finance research and simulation project focused on stochastic price modeling, Monte Carlo option pricing, statistical diagnostics, and financial risk analysis.

The project explores how different assumptions about returns, volatility, jumps, and tail behavior influence simulated financial outcomes. It combines mathematical modeling, numerical methods, and interactive visualization to support experimentation with quantitative finance concepts.

> **Project focus:** Educational and research-oriented quantitative finance—not production trading or investment advice.

---

## Overview

Monte Carlo methods are widely used in quantitative finance to estimate the value and risk of financial instruments under uncertainty.

This project implements and compares multiple stochastic models, pricing approaches, and risk-analysis techniques to study:

- Asset-price dynamics
- Heavy-tailed return behavior
- Sudden price jumps
- Time-varying volatility
- European option pricing
- Black–Scholes benchmark comparisons
- Value at Risk and Expected Shortfall
- Monte Carlo convergence
- Variance-reduction methods
- Statistical model diagnostics

---

## Core Capabilities

| Area | Capabilities |
|---|---|
| Stochastic Modeling | GBM, Student-t returns, Merton Jump Diffusion, GARCH-based volatility |
| Option Pricing | Monte Carlo pricing, European options, Black–Scholes comparison |
| Risk Analysis | Value at Risk, Expected Shortfall, tail-risk analysis |
| Numerical Methods | Random sampling, quasi-random sampling, convergence analysis |
| Variance Reduction | Antithetic variates, control variates, Sobol sequences |
| Diagnostics | Histograms, Q-Q plots, distribution analysis, model comparison |
| Visualization | Simulated price paths, return distributions, pricing and risk outputs |
| Interface | Interactive dashboard for simulation and model exploration |

---

## Mathematical Models

### Geometric Brownian Motion

Geometric Brownian Motion models an asset price as a continuous diffusion process.

It is commonly expressed as:

\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
\]

where:

- \(S_t\) is the asset price
- \(\mu\) is the drift
- \(\sigma\) is the volatility
- \(W_t\) is a Wiener process

GBM is commonly associated with the Black–Scholes framework and assumes normally distributed log returns under constant parameters.

### Student-t Return Model

The Student-t model allows returns to have heavier tails than a normal distribution.

This makes it useful for studying scenarios involving:

- More frequent extreme returns
- Higher tail risk
- Non-normal return behavior
- Differences between Gaussian and heavy-tailed assumptions

### Merton Jump Diffusion

The Merton Jump Diffusion model extends a continuous diffusion process by adding random discontinuous jumps.

It can represent sudden market movements caused by events such as:

- Unexpected news
- Earnings announcements
- Macroeconomic shocks
- Market dislocations

The model helps illustrate how jumps can change pricing and risk estimates.

### GARCH-Based Volatility

GARCH-style modeling allows volatility to change over time.

It is useful for studying:

- Volatility clustering
- Periods of high and low market uncertainty
- Time-varying risk
- The limitations of constant-volatility assumptions

---

## Option Pricing

The project uses Monte Carlo simulation to estimate option prices by simulating possible future asset-price paths.

The general pricing workflow is:

1. Generate simulated asset-price paths.
2. Calculate the terminal asset price.
3. Calculate the option payoff.
4. Average the simulated payoffs.
5. Discount the expected payoff to the present.

For a European call option, the payoff is:

\[
\max(S_T - K, 0)
\]

where:

- \(S_T\) is the asset price at maturity
- \(K\) is the strike price

The Monte Carlo estimate can be compared with the analytical Black–Scholes price to study:

- Pricing differences
- Sampling error
- Convergence
- Model assumptions
- Numerical stability

---

## Risk Analysis

### Value at Risk

Value at Risk estimates a loss threshold over a specified time horizon and confidence level.

For example, a VaR estimate can describe a loss percentile under the simulated return distribution.

VaR does not describe the average loss beyond that threshold.

### Expected Shortfall

Expected Shortfall estimates the average loss in the tail beyond the selected VaR threshold.

It provides additional information about the severity of extreme losses and is particularly useful when studying heavy-tailed or jump-based models.

### Risk Analysis Focus

The project examines how risk estimates change under different assumptions, including:

- Normal versus heavy-tailed returns
- Continuous diffusion versus jump processes
- Constant versus time-varying volatility
- Different simulation sizes
- Different confidence levels
- Different model parameters

---

## Variance-Reduction Methods

Monte Carlo estimates can contain sampling error. Variance-reduction methods aim to improve simulation efficiency and stability.

### Antithetic Variates

Antithetic variates use negatively related simulation samples to reduce estimator variance.

### Control Variates

Control variates use a related quantity with a known expected value to improve the accuracy of an estimate.

For option pricing, the Black–Scholes framework can serve as a benchmark when the relevant assumptions are satisfied.

### Sobol Sequences

Sobol sequences are low-discrepancy quasi-random sequences designed to cover a sampling space more evenly than purely random samples in suitable settings.

These methods are used to investigate how sampling techniques affect:

- Pricing estimates
- Convergence
- Simulation stability
- Computational efficiency

---

## Statistical Diagnostics

The project includes statistical and numerical diagnostics for examining simulation behavior.

These may include:

- Price-path plots
- Return-distribution histograms
- Q-Q plots
- Model-distribution comparisons
- Tail-behavior analysis
- Convergence plots
- Pricing-error analysis
- Simulation-size comparisons
- Variance-reduction comparisons

Diagnostics are intended to make model assumptions and simulation behavior easier to inspect rather than treating model outputs as unquestionable results.

---

## System Workflow

```text
Select Model
     │
     ▼
Define Parameters
     │
     ▼
Generate Random or Quasi-Random Samples
     │
     ▼
Simulate Asset Prices or Returns
     │
     ▼
Calculate Terminal Prices and Payoffs
     │
     ▼
Estimate Option Prices and Risk Metrics
     │
     ▼
Evaluate Convergence and Diagnostics
     │
     ▼
Compare Models and Assumptions
     │
     ▼
Visualize Results

---

## Project Structure

```text
Quant_Monte_Carlo/
│
├── app/
│   └── dashboard.py
│
├── models/
│   ├── gbm.py
│   ├── student_t.py
│   ├── merton_jump_diffusion.py
│   └── garch.py
│
├── pricing/
│   ├── monte_carlo_pricing.py
│   ├── black_scholes.py
│   └── variance_reduction.py
│
├── risk/
│   ├── var.py
│   ├── expected_shortfall.py
│   └── risk_analysis.py
│
├── diagnostics/
│   ├── convergence.py
│   ├── distributions.py
│   └── model_comparison.py
│
├── data/
├── tests/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

Installation
1. Clone the Repository
git clone https://github.com/Jayanthi427/Quant_Monte_Carlo.git
cd Quant_Monte_Carlo
2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Usage

Run the project using the appropriate entry point.

python app/dashboard.py

If the dashboard uses Streamlit, run:

streamlit run app/dashboard.py

The interface can be used to explore:

Different stochastic models
Simulation parameters
Number of Monte Carlo paths
Time horizons
Volatility assumptions
Option parameters
Risk confidence levels
Variance-reduction techniques
Model-comparison outputs
Example Research Questions

This project can be used to investigate:

How does increasing the number of simulations affect pricing convergence?
How different are Monte Carlo prices from Black–Scholes prices under GBM assumptions?
How do heavy-tailed returns affect VaR and Expected Shortfall?
How do jumps change simulated price distributions?
How does volatility clustering influence risk estimates?
Do antithetic variates reduce pricing-estimator variance?
When do Sobol sequences improve convergence?
How sensitive are option prices to drift, volatility, maturity, and strike?
How do different stochastic assumptions change simulated returns?
What are the limitations of simplified financial models?
Research and Educational Applications

The project supports learning and experimentation in:

Quantitative finance
Financial mathematics
Stochastic processes
Derivatives pricing
Numerical methods
Statistical simulation
Risk management
Computational finance
Probability and statistics
Model validation
Limitations

This project is research-oriented and should not be treated as a production-grade pricing or risk platform.

Important limitations include:

Simplified model assumptions
Parameter sensitivity
Sampling error
Dependence on input-data quality
Limited historical-data validation
Limited out-of-sample testing
Simplified treatment of transaction costs and liquidity
No guarantee that simulated results represent real market behavior
Potential numerical instability under extreme parameter settings

The outputs should be interpreted as model-dependent estimates rather than precise predictions of future market outcomes.

Future Improvements

Potential future improvements include:

Improved historical-data pipelines
More robust parameter calibration
Genuine out-of-sample backtesting
Additional stochastic-volatility models
Heston model implementation
Local-volatility modeling
More advanced jump processes
Greeks estimation
Sensitivity analysis
Better numerical benchmarking
Parallelized simulation
GPU acceleration
More comprehensive automated tests
Improved model-validation workflows
Interactive experiment tracking
Exportable research reports
Expanded asset-class support
Technologies
Python
NumPy
SciPy
Pandas
Matplotlib
Plotly
Streamlit
yFinance
Statistical and numerical computing libraries
Skills Demonstrated

This project demonstrates practical work with:

Stochastic-process modeling
Monte Carlo simulation
Derivatives pricing
Black–Scholes benchmarking
Quantitative risk analysis
Value at Risk and Expected Shortfall
Probability distributions
Numerical methods
Variance-reduction techniques
Statistical diagnostics
Financial data analysis
Interactive data visualization
Research-oriented software development

Disclaimer

This project is intended for educational, research, and experimentation purposes only.
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
It does not provide financial advice, investment recommendations, trading signals, or guaranteed pricing or risk estimates.

Financial models are simplified representations of market behavior. Their outputs depend on assumptions, parameters, data quality, and simulation methodology.
