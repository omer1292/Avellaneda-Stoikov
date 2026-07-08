import numpy as np
import pandas as pd

from model import AvellanedaStoikovMM


def run_backtest(prices, t_grid, gamma, sigma, k, T, A=140.0, seed=None):
    rng = np.random.default_rng(seed)
    mm = AvellanedaStoikovMM(gamma=gamma, sigma=sigma, k=k, T=T, A=A)

    n_steps = len(prices) - 1
    dt = T / n_steps

    inventory = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    bids = np.zeros(n_steps + 1)
    asks = np.zeros(n_steps + 1)
    fills_buy = np.zeros(n_steps + 1, dtype=bool)
    fills_sell = np.zeros(n_steps + 1, dtype=bool)

    for i in range(n_steps):
        s = prices[i]
        t = t_grid[i]
        q = inventory[i]

        bid, ask = mm.quotes(s, q, t)
        bids[i], asks[i] = bid, ask

        dist_bid = max(s - bid, 0.0)
        dist_ask = max(ask - s, 0.0)

        p_buy_fill = mm.fill_probability(dist_bid, dt)
        p_sell_fill = mm.fill_probability(dist_ask, dt)

        buy_filled = rng.random() < p_buy_fill
        sell_filled = rng.random() < p_sell_fill

        new_q = q
        new_cash = cash[i]

        if buy_filled:
            new_q += 1
            new_cash -= bid
            fills_buy[i] = True
        if sell_filled:
            new_q -= 1
            new_cash += ask
            fills_sell[i] = True

        inventory[i + 1] = new_q
        cash[i + 1] = new_cash

    bids[-1], asks[-1] = bids[-2], asks[-2]

    mtm = cash + inventory * prices
    pnl = mtm - mtm[0]

    df = pd.DataFrame({
        "t": t_grid,
        "mid_price": prices,
        "bid": bids,
        "ask": asks,
        "inventory": inventory,
        "cash": cash,
        "mtm_pnl": pnl,
        "buy_fill": fills_buy,
        "sell_fill": fills_sell,
    })
    return df


def run_naive_baseline(prices, t_grid, spread, T, k, A=140.0, seed=None):
    rng = np.random.default_rng(seed)
    n_steps = len(prices) - 1
    dt = T / n_steps

    inventory = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)

    for i in range(n_steps):
        s = prices[i]
        bid = s - spread / 2.0
        ask = s + spread / 2.0

        dist = spread / 2.0
        intensity = A * np.exp(-k * dist)
        p_fill = 1.0 - np.exp(-intensity * dt)

        buy_filled = rng.random() < p_fill
        sell_filled = rng.random() < p_fill

        new_q = inventory[i]
        new_cash = cash[i]
        if buy_filled:
            new_q += 1
            new_cash -= bid
        if sell_filled:
            new_q -= 1
            new_cash += ask

        inventory[i + 1] = new_q
        cash[i + 1] = new_cash

    mtm = cash + inventory * prices
    pnl = mtm - mtm[0]

    df = pd.DataFrame({
        "t": t_grid,
        "mid_price": prices,
        "inventory": inventory,
        "cash": cash,
        "mtm_pnl": pnl,
    })
    return df