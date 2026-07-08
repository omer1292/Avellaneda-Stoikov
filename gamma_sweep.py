import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate_data import generate_gbm_path
from backtest import run_backtest


def run_gamma_sweep(gammas, n_runs=200, T=1.0, n_steps=5000, s0=100.0,
                     sigma=2.0, k=1.5):
    rows = []
    for gamma in gammas:
        pnls, inv_stds, max_invs = [], [], []
        for seed in range(n_runs):
            t, prices = generate_gbm_path(s0=s0, sigma=sigma, T=T,
                                           n_steps=n_steps, seed=seed)
            result = run_backtest(prices, t, gamma=gamma, sigma=sigma,
                                   k=k, T=T, seed=seed + 10000)
            pnls.append(result.mtm_pnl.iloc[-1])
            inv_stds.append(result.inventory.std())
            max_invs.append(result.inventory.abs().max())

        mean_pnl, pnl_std = np.mean(pnls), np.std(pnls)
        sharpe = mean_pnl / pnl_std if pnl_std > 0 else np.nan

        rows.append({
            "gamma": gamma,
            "mean_pnl": mean_pnl,
            "pnl_std": pnl_std,
            "mean_inventory_std": np.mean(inv_stds),
            "mean_max_inventory": np.mean(max_invs),
            "sharpe_like": sharpe,
        })

    return pd.DataFrame(rows)


def plot_gamma_sweep(df, out_path="gamma_sweep.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(df["gamma"], df["mean_inventory_std"], marker="o")
    axes[0].set_xlabel("gamma (risk aversion)")
    axes[0].set_ylabel("mean inventory std dev")
    axes[0].set_title("Inventory risk vs gamma")
    axes[0].set_xscale("log")

    axes[1].plot(df["gamma"], df["pnl_std"], marker="o", color="tab:orange")
    axes[1].set_xlabel("gamma (risk aversion)")
    axes[1].set_ylabel("P&L std dev")
    axes[1].set_title("P&L volatility vs gamma")
    axes[1].set_xscale("log")

    axes[2].plot(df["gamma"], df["sharpe_like"], marker="o", color="tab:green")
    axes[2].set_xlabel("gamma (risk aversion)")
    axes[2].set_ylabel("mean P&L / P&L std dev")
    axes[2].set_title("Risk-adjusted return vs gamma")
    axes[2].set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    gammas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    df = run_gamma_sweep(gammas, n_runs=200)
    print(df.to_string(index=False))
    df.to_csv("gamma_sweep_results.csv", index=False)
    plot_gamma_sweep(df)
    print("\nSaved gamma_sweep_results.csv and gamma_sweep.png")