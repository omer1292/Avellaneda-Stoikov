import numpy as np

from simulate_data import generate_gbm_path
from backtest import run_backtest, run_naive_baseline

# gamma=0.2 chosen from gamma_sweep.py as the best risk-adjusted setting.
# Naive baseline uses the AS model's own t=0 spread as its fixed spread,
# so the comparison isn't stacked by picking an arbitrary naive width.


def main(gamma=0.2, n_runs=200, T=1.0, n_steps=5000, s0=100.0, sigma=2.0, k=1.5):
    as_pnls, as_inv_stds, as_max_inv = [], [], []
    naive_pnls, naive_inv_stds, naive_max_inv = [], [], []

    from model import AvellanedaStoikovMM
    mm = AvellanedaStoikovMM(gamma=gamma, sigma=sigma, k=k, T=T)
    naive_spread = mm.optimal_spread(t=0)

    for seed in range(n_runs):
        t, prices = generate_gbm_path(s0=s0, sigma=sigma, T=T, n_steps=n_steps, seed=seed)

        as_result = run_backtest(prices, t, gamma=gamma, sigma=sigma, k=k, T=T, seed=seed + 5000)
        naive_result = run_naive_baseline(prices, t, spread=naive_spread, T=T, k=k, seed=seed + 5000)

        as_pnls.append(as_result.mtm_pnl.iloc[-1])
        as_inv_stds.append(as_result.inventory.std())
        as_max_inv.append(as_result.inventory.abs().max())

        naive_pnls.append(naive_result.mtm_pnl.iloc[-1])
        naive_inv_stds.append(naive_result.inventory.std())
        naive_max_inv.append(naive_result.inventory.abs().max())

    as_mean_pnl, as_pnl_std = np.mean(as_pnls), np.std(as_pnls)
    naive_mean_pnl, naive_pnl_std = np.mean(naive_pnls), np.std(naive_pnls)
    as_inv_std_mean = np.mean(as_inv_stds)
    naive_inv_std_mean = np.mean(naive_inv_stds)

    print(f"gamma={gamma}, n_runs={n_runs}, fixed spread used for naive: {naive_spread:.4f}\n")

    print(f"{'metric':>25} {'AS':>15} {'naive':>15}")
    print(f"{'mean P&L':>25} {as_mean_pnl:>15.2f} {naive_mean_pnl:>15.2f}")
    print(f"{'P&L std dev':>25} {as_pnl_std:>15.2f} {naive_pnl_std:>15.2f}")
    print(f"{'mean inventory std':>25} {as_inv_std_mean:>15.3f} {naive_inv_std_mean:>15.3f}")
    print(f"{'mean max inventory':>25} {np.mean(as_max_inv):>15.2f} {np.mean(naive_max_inv):>15.2f}")

    pnl_std_reduction = 1 - (as_pnl_std / naive_pnl_std)
    inv_std_reduction = 1 - (as_inv_std_mean / naive_inv_std_mean)
    pnl_change = (as_mean_pnl - naive_mean_pnl) / abs(naive_mean_pnl)

    print(f"\nAS P&L std dev is {pnl_std_reduction*100:.1f}% lower than naive")
    print(f"AS mean inventory std dev is {inv_std_reduction*100:.1f}% lower than naive")
    print(f"AS mean P&L is {pnl_change*100:.1f}% {'higher' if pnl_change > 0 else 'lower'} than naive")

    as_sharpe = as_mean_pnl / as_pnl_std if as_pnl_std > 0 else float('nan')
    naive_sharpe = naive_mean_pnl / naive_pnl_std if naive_pnl_std > 0 else float('nan')
    print(f"\nAS Sharpe-like ratio: {as_sharpe:.4f}")
    print(f"naive Sharpe-like ratio: {naive_sharpe:.4f}")


if __name__ == "__main__":
    main()