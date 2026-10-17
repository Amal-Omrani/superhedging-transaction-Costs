# Explicit super-replication prices under proportional transaction costs: numerical experiments

Code for the numerical section of

> E. Lepinette and A. Omrani, *Explicit Recursive Construction of Super-Replication Prices under Proportional
> Transaction Costs*. Preprint: [arXiv:2503.02419](https://doi.org/10.48550/arXiv.2503.02419).

The repository contains the code that produced the numerical results of the paper (SPY, European call, 100 weeks,
proportional cost rates from 0.2% to 2%).

## Contents

| File | Role |
|---|---|
| `superhedging.py` | The method of the paper: one-step backward recursion (Cases 1 and 2), AIP check, optimal strategies. |
| `spy_experiment.py` | The SPY experiment of the paper: data, weekly backtest, Table 1 (`performance_stats.tex`) and figures. |
| `results/main/` | Outputs of `spy_experiment.py`. |

## Installation and use

Python 3.13. The versions with which the results were produced are pinned in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd results/main && python ../../spy_experiment.py   # Table 1 and figures, about 20 s
```

The data are downloaded from Yahoo Finance at run time (`SPY`, daily closes, 3 June 2013 to 29 December 2016,
903 observations). With `yfinance` 0.2.66 these are dividend-adjusted closes. Yahoo recomputes the adjustment
after each new dividend, so the price levels obtained today differ from those of an earlier download by a constant
factor; the returns, hence all the reported results (expressed relative to the spot), are unaffected.

## Design of the experiment (as implemented in `spy_experiment.py`)

* **Weeks.** Week `j = 0, ..., 99` consists of the four consecutive trading days starting at day
  `260 + 4 (j - 1)`, so that `T = 3` and the portfolio is rebalanced at `t = 0, 1, 2`.
* **Conditional supports.** For each transition `t -> t+1`, `t = 0, 1, 2`, the support
  `[alpha_t, beta_t]` of `S_{t+1}/S_t` is the minimum and the maximum of the corresponding daily ratios over
  the first 260 trading days of the sample (65 blocks of four days); the same supports are used for every week.
* **Claim.** At-the-money call, `K = S_0`.
* **Algorithm.** Backward recursion of the paper from `g_3(x) = (x - K)^+`; price `V_0 = g_0(0, S_0)`;
  optimal strategy computed forward along the realized path; self-financing portfolio with costs
  `kappa |phi_t - phi_{t-1}| S_t`.
* **Output.** Relative error `eps = (V_3 - (S_3 - K)^+) / S_0` and price `V_0 / S_0`, for
  `kappa = 0.2%, 0.4%, ..., 2%`.

Running time: about 20 ms per week and cost rate on a laptop (Apple silicon).

## Results

`results/main/performance_stats.tex` reproduces Table 1 of the paper:

| kappa (%) | mean eps (%) | std eps (%) | V0/S0 (%) | P(eps >= 0) |
|---|---|---|---|---|
| 0.2 | 1.36 | 0.53 | 2.07 | 0.97 |
| 1.0 | 1.56 | 0.61 | 2.78 | 0.98 |
| 2.0 | 1.84 | 0.71 | 3.71 | 0.98 |

The last block of `spy_experiment.py` reports the fraction of daily ratios outside the calibrated supports, 29 of 300
(9.67%), over weeks `j = 1, ..., 100`; over the 100 weeks of the table (`j = 0, ..., 99`) it is 28 of 300 (9.33%).

## License

MIT (see `LICENSE`).
