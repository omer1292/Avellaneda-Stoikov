import requests
import pandas as pd
import time

# Binance public trade endpoint, max 1000 trades per call, no API key needed.
def fetch_recent_trades(symbol="BTCUSDT", limit=1000):
    url = "https://api.binance.com/api/v3/trades"
    params = {"symbol": symbol, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "price", "qty", "isBuyerMaker"]]


# Polls repeatedly to build a longer window than the 1000-trade cap allows.
def poll_trades_over_window(symbol="BTCUSDT", n_polls=20, sleep_seconds=30):
    all_trades = []
    for i in range(n_polls):
        df = fetch_recent_trades(symbol=symbol)
        all_trades.append(df)
        print(f"poll {i+1}/{n_polls}: got {len(df)} trades, "
              f"latest price {df['price'].iloc[-1]}")
        if i < n_polls - 1:
            time.sleep(sleep_seconds)

    combined = pd.concat(all_trades).drop_duplicates(subset=["time", "price", "qty"])
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


if __name__ == "__main__":
    trades = fetch_recent_trades(symbol="BTCUSDT")
    print(trades.head())
    print(f"\nGot {len(trades)} trades spanning "
          f"{trades['time'].iloc[0]} to {trades['time'].iloc[-1]}")

    trades.to_csv("btc_trades_snapshot.csv", index=False)
    print("\nSaved to btc_trades_snapshot.csv")

    trades = poll_trades_over_window(symbol="BTCUSDT", n_polls=40, sleep_seconds=30)
    trades.to_csv("btc_trades_window.csv", index=False)