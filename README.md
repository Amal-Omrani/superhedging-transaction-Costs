# Explicit super-replication prices under proportional transaction costs: numerical experiments

Code for the numerical section of:

> E. Lepinette and A. Omrani, *Explicit Recursive Construction of Super-Replication Prices under Proportional Transaction Costs*. Preprint: [arXiv:2503.02419](https://doi.org/10.48550/arXiv.2503.02419).

The repository implements the paper's explicit backward recursion and optimal-strategy construction, and applies them to a rolling SPY backtest with proportional transaction costs. It also generates a sensitivity analysis for the calibrated supports and transaction-cost parameter.

## Repository contents

| File | Description |
|---|---|
| `superhedging.py` | Core implementation of the one-step backward recursion, including Cases 1 and 2, the Arbitrage-free Interval Property (AIP) check, payoff-function cleaning, optimal strategies, portfolio values, and superhedging errors. |
| `spy_experiment.py` | End-to-end SPY experiment: downloads historical prices, calibrates conditional supports, prices an at-the-money European call, computes optimal strategies, runs the weekly backtest, writes summary statistics, creates plots, and checks support violations. |
| `performance_stats.tex` | LaTeX table of the generated performance statistics for all transaction-cost levels. |
| `requirements.txt` | Pinned Python dependencies used to produce the numerical results. |
| `run.log` | Example console output from an experiment run. |
| `V0_percent_vs_kappa.png` | Mean relative initial superhedging price, with error bars, versus the transaction-cost rate. |
| `V0_vs_kappa.png` | Superhedging price `V0` versus the transaction-cost rate. |
| `superhedging_error_vs_kappa.png` | Mean superhedging error versus the transaction-cost rate. |
| `boxplot_superhedging_error_vs_kappa.png` | Distribution of superhedging errors by transaction-cost rate. |
| `kde_superhedging_error_vs_kappa.png` | Kernel-density estimates of superhedging errors by transaction-cost rate. |
| `LICENSE` | MIT license. |

## Installation and usage

The results were produced with Python 3.13 and the versions pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python spy_experiment.py
```

The script downloads SPY data from Yahoo Finance at runtime. It uses daily closing prices from June 3, 2013 through December 29, 2016, giving 903 observations and 902 return observations for the downloaded sample. The generated LaTeX table and figures are written to the repository root, replacing files with the same names.

> **Note:** `spy_experiment.py` executes the experiment at import/run time; it is intended to be run as a script from the repository root.

## Experiment design

The implementation in `spy_experiment.py` uses the following setup:

- **Underlying:** SPY, downloaded with `yfinance`.
- **Calibration window:** 260 trading days.
- **Backtest horizon:** 100 four-observation blocks, with three rebalancing periods per block.
- **Conditional supports:** for each within-block transition, the minimum and maximum historical daily price ratios are computed from the calibration window and used as the support bounds.
- **Claim:** an at-the-money European call, with strike `K = S_0` for each block.
- **Recursion:** starts from the terminal payoff `g_T(x) = (x - K)^+` and applies the paper's backward recursion at each time step.
- **Strategy:** the corresponding optimal strategy is computed forward along each realized SPY path.
- **Transaction costs:** proportional costs of `kappa |theta_t - theta_{t-1}| S_t`.
- **Moderate cost grid:** `kappa = 0.2%, 0.4%, ..., 2.0%`.
- **Optional extended grid:** the script also defines `kappa = 0.3, 0.4, 0.5`; select that grid in the source if desired.

The script first runs a demonstration for week 1 at `kappa = 1%`, then runs the full grid of transaction-cost rates over the 100 blocks. It reports the initial superhedging price, optimal holdings, final portfolio value, and relative superhedging error.

## Reported outputs

`performance_stats.tex` contains the following columns for each transaction-cost rate:

- mean relative superhedging error, in percent;
- standard deviation of the relative error, in percent;
- coefficient of variation;
- mean `V_0/S_0`, in percent; and
- the fraction of simulations with non-negative superhedging error.

The checked-in table reports:

| `kappa` (%) | Mean error (%) | Std. dev. (%) | CV | `V0/S0` (%) | `P(error >= 0)` |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 1.36 | 0.53 | 0.39 | 2.07 | 0.97 |
| 0.40 | 1.41 | 0.55 | 0.39 | 2.24 | 0.98 |
| 0.60 | 1.45 | 0.57 | 0.39 | 2.42 | 0.98 |
| 0.80 | 1.52 | 0.59 | 0.39 | 2.61 | 0.98 |
| 1.00 | 1.56 | 0.61 | 0.39 | 2.78 | 0.98 |
| 1.20 | 1.58 | 0.67 | 0.42 | 2.93 | 0.97 |
| 1.40 | 1.71 | 0.66 | 0.39 | 3.08 | 0.98 |
| 1.60 | 1.77 | 0.68 | 0.38 | 3.30 | 0.98 |
| 1.80 | 1.80 | 0.69 | 0.38 | 3.51 | 0.98 |
| 2.00 | 1.84 | 0.71 | 0.39 | 3.71 | 0.98 |

The support-violation check at the end of `spy_experiment.py` compares realized prices with the calibrated support intervals over the 100 backtest blocks and prints the number and fraction of observations outside those intervals.

## Reproducibility notes

- Yahoo Finance data are downloaded when the script runs, so results may vary if the provider revises historical prices or if the data-download behavior changes.
- The script currently labels prices with a currency symbol in console output, but the downloaded SPY series is used as returned by Yahoo Finance.
- The generated plots and `performance_stats.tex` are repository-root outputs; no `results/main/` directory is required.

## License

MIT License; see [`LICENSE`](LICENSE).
