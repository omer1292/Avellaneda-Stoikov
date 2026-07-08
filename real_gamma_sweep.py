import numpy as np
import pandas as pd

from estimate_params import load_trades, estimate_sigma_from_trades, estimate_k_from_trades
from run_real_backtest import build_price_path
from backtest import run_backtest

# T is in real seconds here , so gamma needs to be much 
# smaller than the 0.01-1.0 range used in synthetic tests where T=1.0.


def sweep(csv_path="btc_trades_window.csv", gammas=None, bar_seconds=1, seed=42):
    if gammas is None:
        gammas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

    df = load_trades(csv_path)
    sigma_frac, _ = estimate_sigma_from_trades(df)
    A_fit, k_fit, _ = estimate_k_from_trades(df)

    t_grid, prices = build_price_path(df, bar_seconds=bar_seconds)
    price_level = prices[0]
    sigma_abs = price_level * sigma_frac
    T = t_grid[-1]

    print(f"T={T:.0f}s, sigma_abs={sigma_abs:.4f}, k={k_fit:.4f}, A={A_fit:.4f}\n")
    print(f"{'gamma':>10} {'initial_spread':>16} {'final_pnl':>12} {'inv_std':>10} {'max_inv':>10} {'n_fills':>10}")

    rows = []
    for gamma in gammas:
        result = run_backtest(prices, t_grid, gamma=gamma, sigma=sigma_abs,
                               k=k_fit, T=T, A=A_fit, seed=seed)
        n_fills = int(result.buy_fill.sum() + result.sell_fill.sum())
        initial_spread = result.ask.iloc[0] - result.bid.iloc[0]
        final_pnl = result.mtm_pnl.iloc[-1]
        inv_std = result.inventory.std()
        max_inv = result.inventory.abs().max()

        print(f"{gamma:>10} {initial_spread:>16.2f} {final_pnl:>12.2f} {inv_std:>10.3f} {max_inv:>10.1f} {n_fills:>10}")
        rows.append({"gamma": gamma, "initial_spread": initial_spread,
                      "final_pnl": final_pnl, "inv_std": inv_std,
                      "max_inv": max_inv, "n_fills": n_fills})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "btc_trades_window.csv"
    sweep(csv_path)