import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from simulate_data import estimate_volatility

# k is approximated from trade prints only (no L2 order book data).


def load_trades(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def estimate_sigma_from_trades(df):
    df = df.set_index("time")
    price_series = df["price"].resample("1s").last().ffill()
    sigma_per_second = estimate_volatility(price_series.values, dt=1.0)
    return sigma_per_second, price_series


def estimate_k_from_trades(df, local_mid_window="5s", n_bins=15):
    df = df.copy().set_index("time")
    local_mid = df["price"].rolling(local_mid_window).mean()
    df["distance"] = (df["price"] - local_mid).abs()
    df = df.dropna(subset=["distance"])

    if len(df) < 20:
        raise ValueError(
            "Not enough trades after computing rolling mid-price. "
            "Pull more data (increase n_polls) or shorten local_mid_window."
        )

    max_dist = df["distance"].quantile(0.95)
    bins = np.linspace(0, max_dist, n_bins + 1)
    df["dist_bin"] = pd.cut(df["distance"], bins=bins)

    total_time_seconds = (df.index.max() - df.index.min()).total_seconds()
    counts = df.groupby("dist_bin", observed=True).size()
    bin_centers = np.array([interval.mid for interval in counts.index])
    intensities = counts.values / total_time_seconds

    valid = intensities > 0
    bin_centers = bin_centers[valid]
    intensities = intensities[valid]

    def exp_decay(delta, A, k):
        return A * np.exp(-k * delta)

    p0 = [intensities[0] if len(intensities) > 0 else 1.0, 1.0]
    try:
        popt, _ = curve_fit(exp_decay, bin_centers, intensities, p0=p0, maxfev=5000)
        A_fit, k_fit = popt
    except RuntimeError:
        A_fit, k_fit = np.nan, np.nan

    fit_data = pd.DataFrame({"distance": bin_centers, "intensity": intensities})
    return A_fit, k_fit, fit_data


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "btc_trades_window.csv"
    print(f"Loading {csv_path}...")
    df = load_trades(csv_path)
    print(f"Loaded {len(df)} trades spanning "
          f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    sigma, price_series = estimate_sigma_from_trades(df)
    print(f"\nEstimated sigma (per second): {sigma:.6f}")

    try:
        A_fit, k_fit, fit_data = estimate_k_from_trades(df)
        print(f"\nEstimated A (base arrival intensity): {A_fit:.4f}")
        print(f"Estimated k (decay rate): {k_fit:.4f}")
        print("\nFit data (distance vs intensity):")
        print(fit_data.to_string(index=False))
    except ValueError as e:
        print(f"\nCould not estimate k: {e}")