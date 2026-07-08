import numpy as np

# Simulate a GBM mid-price path, returns (t, s).
def generate_gbm_path(s0, sigma, T, n_steps, mu=0.0, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    shocks = rng.normal(loc=0.0, scale=1.0, size=n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shocks

    log_prices = np.log(s0) + np.concatenate([[0.0], np.cumsum(log_returns)])
    s = np.exp(log_prices)

    return t, s


# Realized vol of log returns, scaled to the same time unit as dt.
def estimate_volatility(prices, dt):
    log_returns = np.diff(np.log(prices))
    return np.std(log_returns) / np.sqrt(dt)