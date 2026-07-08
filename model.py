# Avellaneda-Stoikov market making model.
# Ref: Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224.
# Quotes are centered on a reservation price (not raw mid) that shifts with
# inventory, skewing quotes to pull inventory back toward zero.

import numpy as np


class AvellanedaStoikovMM:
    def __init__(self, gamma, sigma, k, T, A=140.0):
        self.gamma = gamma  # risk aversion
        self.sigma = sigma  # volatility
        self.k = k          # order arrival decay
        self.T = T          # session length
        self.A = A          # base arrival intensity (fill sim only)

    def reservation_price(self, s, q, t):
        # r = s - q * gamma * sigma^2 * (T - t)
        return s - q * self.gamma * (self.sigma ** 2) * (self.T - t)

    def optimal_spread(self, t):
        # spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        risk_term = self.gamma * (self.sigma ** 2) * (self.T - t)
        fill_term = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        return risk_term + fill_term

    def quotes(self, s, q, t):
        r = self.reservation_price(s, q, t)
        spread = self.optimal_spread(t)
        bid = r - spread / 2.0
        ask = r + spread / 2.0
        return bid, ask

    def fill_probability(self, distance_from_mid, dt):
        # Poisson fill prob: lambda(delta) = A * exp(-k * delta)
        intensity = self.A * np.exp(-self.k * distance_from_mid)
        return 1.0 - np.exp(-intensity * dt)