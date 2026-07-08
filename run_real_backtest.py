import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import AvellanedaStoikovMM
from estimate_params import load_trades, estimate_sigma_from_trades, estimate_k_from_trades
from backtest import run_backtest, run_naive_baseline

# sigma must be in absolute price units, not fractional. estimate_params
# returns fractional sigma, converted below via sigma_abs = price * sigma_frac.


def build_price_path(df, bar_seconds=1):
    df = df.set_index("time")
    price_series = df["price"].resample(f"{bar_seconds}s").last().ffill()
    price_series = price_series.dropna()

    prices = price_series.values
    t_grid = np.arange(len(prices)) * bar_seconds
    return t_grid.astype(float), prices


def main(csv_path="btc_trades_window.csv", gamma=0.0001, bar_seconds=1, seed=42):
    print(f"Loading {csv_path}...")
    df = load_trades(csv_path)
    print(f"Loaded {len(df)} trades")

    sigma_frac, _ = estimate_sigma_from_trades(df)
    A_fit, k_fit, fit_data = estimate_k_from_trades(df)

    t_grid, prices = build_price_path(df, bar_seconds=bar_seconds)
    price_level = prices[0]
    sigma_abs = price_level * sigma_frac

    T = t_grid[-1]
    print(f"\nSession length T = {T:.0f} seconds ({len(prices)} bars)")
    print(f"Price level: {price_level:.2f}")
    print(f"sigma (fractional): {sigma_frac:.6f}")
    print(f"sigma (absolute, $/sqrt(sec)): {sigma_abs:.4f}")
    print(f"k (fitted): {k_fit:.4f}")
    print(f"A (fitted): {A_fit:.4f}")

    if not np.isfinite(k_fit) or k_fit <= 0:
        print("\nWARNING: k fit failed or non-positive, falling back to k=1.0")
        k_fit = 1.0
    if not np.isfinite(A_fit) or A_fit <= 0:
        print("WARNING: A fit failed or non-positive, falling back to A=1.0")
        A_fit = 1.0

    print(f"\nRunning backtest with gamma={gamma}...")
    as_result = run_backtest(prices, t_grid, gamma=gamma, sigma=sigma_abs,
                              k=k_fit, T=T, A=A_fit, seed=seed)

    naive_spread = 2 * sigma_abs
    naive_result = run_naive_baseline(prices, t_grid, spread=naive_spread,
                                       T=T, k=k_fit, A=A_fit, seed=seed)

    print(f"\n=== Avellaneda-Stoikov (real data) ===")
    print(f"Final P&L: {as_result.mtm_pnl.iloc[-1]:.2f}")
    print(f"Inventory std dev: {as_result.inventory.std():.3f}")
    print(f"Max abs inventory: {as_result.inventory.abs().max():.1f}")

    print(f"\n=== Naive fixed-spread baseline (real data) ===")
    print(f"Final P&L: {naive_result.mtm_pnl.iloc[-1]:.2f}")
    print(f"Inventory std dev: {naive_result.inventory.std():.3f}")
    print(f"Max abs inventory: {naive_result.inventory.abs().max():.1f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(as_result.t, as_result.mid_price, label="mid price", color="black", linewidth=0.8)
    axes[0].plot(as_result.t, as_result.bid, label="AS bid", color="tab:blue", alpha=0.5, linewidth=0.6)
    axes[0].plot(as_result.t, as_result.ask, label="AS ask", color="tab:red", alpha=0.5, linewidth=0.6)
    axes[0].set_ylabel("price ($)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Real BTCUSDT price path with Avellaneda-Stoikov quotes")

    axes[1].plot(as_result.t, as_result.inventory, label="AS inventory", color="tab:blue")
    axes[1].plot(naive_result.t, naive_result.inventory, label="naive inventory", color="tab:gray", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("inventory")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_title("Inventory over time: AS vs naive")

    axes[2].plot(as_result.t, as_result.mtm_pnl, label="AS P&L", color="tab:green")
    axes[2].plot(naive_result.t, naive_result.mtm_pnl, label="naive P&L", color="tab:gray", alpha=0.7)
    axes[2].set_ylabel("mark-to-market P&L ($)")
    axes[2].set_xlabel("time (seconds)")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_title("P&L over time: AS vs naive")

    fig.tight_layout()
    fig.savefig("real_data_backtest.png", dpi=150)
    print("\nSaved plot to real_data_backtest.png")

    as_result.to_csv("as_result_real.csv", index=False)
    naive_result.to_csv("naive_result_real.csv", index=False)
    print("Saved as_result_real.csv and naive_result_real.csv")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "btc_trades_window.csv"
    main(csv_path=csv_path)