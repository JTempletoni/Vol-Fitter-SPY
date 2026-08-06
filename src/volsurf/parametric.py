"""SVI parametric volatility surface (Phase 1)."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, lsq_linear


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_w(k, p):
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma ** 2))


def svi_dw_dk(k, p):
    return p.b * (p.rho + (k - p.m) / np.sqrt((k - p.m) ** 2 + p.sigma ** 2))


def svi_d2w_dk2(k, p):
    return p.b * p.sigma ** 2 / ((k - p.m) ** 2 + p.sigma ** 2) ** 1.5


def svi_iv(k, T, p):
    return np.sqrt(svi_w(k, p) / T)


def durrleman_g(k, p):
    w = svi_w(k, p); wp = svi_dw_dk(k, p); wpp = svi_d2w_dk2(k, p)
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


def risk_neutral_density(k, p):
    w = svi_w(k, p); g = durrleman_g(k, p)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g * np.exp(-d2 ** 2 / 2) / np.sqrt(2 * np.pi * w)


def _initial_guess(k, w):
    i_min = int(np.argmin(w))
    m0 = float(k[i_min])
    sigma0 = max(0.05, (k.max() - k.min()) / 4)
    return m0, sigma0


def fit_svi_slice(k, w_market, T, weights=None, initial=None,
                  m_bounds=(-1.0, 1.0), sigma_bounds=(0.005, 2.0)):
    """Zeliade quasi-explicit SVI calibration (De Marco and Martini 2009)."""
    k = np.asarray(k, dtype=float)
    w_market = np.asarray(w_market, dtype=float)
    weights = np.ones_like(w_market) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(weights)

    def inner_solve(m, sigma):
        y = (k - m) / sigma
        A = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1)])
        bounds = (np.array([-np.inf, -np.inf, 0.0]),
                  np.array([np.inf, np.inf, 4.0 * sigma]))
        try:
            res = lsq_linear(sqrt_w[:, None] * A, sqrt_w * w_market,
                             bounds=bounds, method='trf', max_iter=200)
            a, d, c = res.x
        except Exception:
            return np.inf, 0.0, 0.0, 0.0
        b = c / sigma
        rho = d / c if abs(c) > 1e-12 else 0.0
        rho = float(np.clip(rho, -0.999, 0.999))
        w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        return float(np.sum(weights * (w_model - w_market) ** 2)), a, b, rho

    def objective(x):
        m, sigma = x
        if not (m_bounds[0] <= m <= m_bounds[1] and sigma_bounds[0] <= sigma <= sigma_bounds[1]):
            return 1e10
        rss, *_ = inner_solve(m, sigma)
        return rss

    m0, sigma0 = _initial_guess(k, w_market) if initial is None else (initial.m, initial.sigma)
    res = minimize(objective, [m0, sigma0], method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-12, 'maxiter': 500})
    m_opt, sigma_opt = res.x
    sigma_opt = float(np.clip(sigma_opt, sigma_bounds[0], sigma_bounds[1]))
    rss, a, b, rho = inner_solve(m_opt, sigma_opt)
    return SVIParams(a=float(a), b=float(b), rho=float(rho),
                     m=float(m_opt), sigma=float(sigma_opt)), rss


class SVISurface:
    """Per-expiry SVI fits with linear interpolation in total variance across T."""

    def __init__(self, fits_by_expiry, S, snapshot_date):
        self.fits = fits_by_expiry
        self.S = S
        self.snapshot_date = snapshot_date
        self._sorted = sorted(fits_by_expiry.items(), key=lambda kv: kv[1]['T'])
        self._Ts = np.array([f['T'] for _, f in self._sorted])
        self._params = [f['params'] for _, f in self._sorted]
        self._forwards = [f['forward'] for _, f in self._sorted]

    def total_variance(self, k, T):
        k = np.asarray(k, dtype=float)
        if T <= self._Ts[0]:
            return svi_w(k, self._params[0]) * T / self._Ts[0]
        if T >= self._Ts[-1]:
            return svi_w(k, self._params[-1]) * T / self._Ts[-1]
        i = int(np.searchsorted(self._Ts, T) - 1)
        w_lo = svi_w(k, self._params[i])
        w_hi = svi_w(k, self._params[i + 1])
        t = (T - self._Ts[i]) / (self._Ts[i + 1] - self._Ts[i])
        return (1 - t) * w_lo + t * w_hi

    def iv(self, k, T):
        return np.sqrt(self.total_variance(k, T) / T)

    def density(self, k, T):
        i = int(np.argmin(np.abs(self._Ts - T)))
        return risk_neutral_density(k, self._params[i])

    def __repr__(self):
        return (f'<SVISurface date={self.snapshot_date} S={self.S:.2f} '
                f'slices={len(self._Ts)} T=[{self._Ts.min():.3f}, {self._Ts.max():.3f}]>')
