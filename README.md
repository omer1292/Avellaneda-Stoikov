# Avellaneda-Stoikov Market Making Model

Implementation and backtest of the Avellaneda-Stoikov (2008) market making
model. Started on synthetic price paths, then tested against real BTCUSDT
trade data pulled from Binance.

## The idea

A market maker profits from quoting a bid/ask around fair value and
capturing the spread, but takes on risk from inventory building up when
order flow is one-sided. Avellaneda-Stoikov solves for quotes that account
for this by centering on a "reservation price" that shifts away from
mid-price based on current inventory, instead of quoting symmetrically.

Reservation price: r = s - q·γ·σ²·(T-t)
Optimal spread: spread = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)
Quotes: bid = r - spread/2, ask = r + spread/2

s = mid-price, q = inventory, γ = risk aversion, σ = volatility, T-t =
time remaining, k = how fast fill probability decays with distance from mid.

## Files

| File | Purpose |
|---|---|
| `model.py` | Core `AvellanedaStoikovMM` class |
| `simulate_data.py` | Synthetic GBM price paths, for validating the model before using real data |
| `backtest.py` | Simulates fills, tracks inventory/cash/P&L, plus a naive fixed-spread baseline |
| `gamma_sweep.py` | Risk-aversion sweep on synthetic data (200 runs per γ) |
| `pull_binance_data.py` | Pulls real BTCUSDT trades from Binance's REST API |
| `estimate_params.py` | Estimates σ and k from real trade data |
| `run_real_backtest.py` | Backtest against real data with correctly-scaled params |
| `real_gamma_sweep.py` | γ calibration sweep for real data's time scale |

## What I did and found

First validated the formulas on synthetic GBM paths: at zero inventory the
reservation price equals mid-price, long inventory shifts it down
(encouraging selling), short inventory shifts it up, and spread narrows as
time-to-horizon shrinks. All matched what the theory predicts.

Then ran 200 Monte Carlo runs per γ against a naive fixed-spread baseline
with no inventory skew, to get a real comparison instead of judging off one
noisy path:

| γ | Mean P&L | P&L std dev | Mean inventory std dev | Sharpe-like |
|---|---|---|---|---|
| 0.01 | -213.5 | 3121.5 | 2.83 | -0.068 |
| 0.05 | 30.7 | 866.5 | 1.89 | 0.035 |
| 0.10 | 36.7 | 660.6 | 1.49 | 0.056 |
| 0.20 | 50.6 | 584.6 | 1.15 | 0.087 |
| 0.50 | 38.2 | 386.8 | 0.79 | 0.099 |
| 1.00 | 67.1 | 632.2 | 0.59 | 0.106 |

Higher γ trades some raw P&L for much lower inventory and P&L volatility.
The naive baseline gets higher raw mean P&L (670 vs 264 at comparable
settings) but with 3x the variance, which is basically the whole point of
the model working as intended.

For real data, pulled 6,938 BTCUSDT trades over 575 seconds from Binance
(polled repeatedly since the free endpoint caps at 1,000 trades/call).
Estimated σ from 1-second resampled prices, converted from fractional to
absolute dollar terms since the model needs volatility in the same units
as k and price distance. Estimated k and A by bucketing trades by distance
from a rolling local mid-price and fitting exponential decay, a proxy for
fill intensity since I only had trade prints, not the full order book.

Ran into a real calibration issue here: γ values tuned on synthetic data
(T=1.0) were way off on real data, because T there is 575 real seconds,
and the risk term scales with γ·σ²·(T-t). Had to re-sweep γ specifically
for the real time scale:

| γ | Initial spread ($) | Final P&L | Inventory std | Max inventory | Fills |
|---|---|---|---|---|---|
| 0.05 | 219.21 | 101.80 | 0.17 | 1 | 25 |
| 0.01 | 50.91 | 834.74 | 0.44 | 2 | 170 |
| 0.005 | 29.88 | 1359.24 | 0.58 | 2 | 286 |
| 0.001 | 13.06 | 2395.47 | 1.10 | 3 | 500 |
| 0.0005 | 10.96 | 2641.71 | 1.51 | 5 | 532 |
| **0.0001** | **9.27** | **2790.49** | **2.90** | **8** | 551 |
| 0.00005 | 9.06 | 2784.24 | 3.29 | 8 | 554 |
| 0.00001 | 8.90 | 2631.92 | 3.93 | 11 | 563 |

At γ=0.0001, AS beat the naive baseline on both P&L (2,790 vs 2,270) and
risk (inventory std 2.90 vs 6.61, max 8 vs 15) on this window.

## Limitations

- One 10-minute window isn't enough to claim statistical significance on
  its own, the synthetic 200-run sweep is the actual evidence here. The
  real-data run is more of a concrete illustration.
- k comes from trade prints, not L2 order book depth, so it's an
  approximation of fill intensity rather than a direct measurement.
- The model assumes a hard end-of-session T, which fits an equity market
  maker who needs to be flat by close. Crypto trades 24/7 with no natural
  T, so the spread narrowing near the end of my window is an artifact of
  an arbitrary session length rather than real market structure. A
  rolling-horizon variant (Guéant-Lehalle-Fernandez-Tapia) would fit
  better.
- Fill simulation is Poisson based on distance from mid only, not a full
  order book replay, so it ignores queue position.

## Running it

```bash
pip install numpy pandas matplotlib scipy requests

python3 gamma_sweep.py                          # synthetic validation + sweep
python3 pull_binance_data.py                     # pulls real trades
python3 estimate_params.py btc_trades_window.csv
python3 real_gamma_sweep.py btc_trades_window.csv
python3 run_real_backtest.py btc_trades_window.csv
```
