"""SSVI and xSSVI parametric volatility surfaces (Phase 2).

SSVI: Gatheral-Jacquier (2014). Three parameters (rho, eta, gamma) plus an
ATM total variance term structure theta_T inferred from the data. Calendar
arbitrage-free by construction (theta monotone in T). Butterfly arbitrage-
free under explicit parameter constraints enforced via SLSQP.

xSSVI: per-maturity rho with shared (eta, gamma). Captures the documented
term structure of equity skew. Same calendar and butterfly arbitrage
guarantees via vector-valued SLSQP constraints across adjacent slices.

References:
    Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility
    surfaces. Quantitative Finance 14(1), 59-71.
    Hendriks, S. and Martini, C. (2017). The extended SSVI volatility
    surface. SSRN 2971502.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from .pricing import bs_vega


# ----------------------------------------------------------------
# SSVI parameterisation
# ----------------------------------------------------------------

@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float


def phi_power_law(theta, eta, gamma):
    return eta * np.power(theta, -gamma)


def ssvi_w(k, theta, p):
    phi = phi_power_law(theta, p.eta, p.gamma)
    z = phi * k
    return 0.5 * theta * (1.0 + p.rho * z + np.sqrt((z + p.rho) ** 2 + (1.0 - p.rho ** 2)))


def ssvi_dw_dk(k, theta, p):
    phi = phi_power_law(theta, p.eta, p.gamma)
    z = phi * k
    inner_root = np.sqrt((z + p.rho) ** 2 + (1.0 - p.rho ** 2))
    return 0.5 * theta * phi * (p.rho + (z + p.rho) / inner_root)


def ssvi_d2w_dk2(k, theta, p):
    phi = phi_power_law(theta, p.eta, p.gamma)
    z = phi * k
    denom = np.power((z + p.rho) ** 2 + (1.0 - p.rho ** 2), 1.5)
    return 0.5 * theta * (phi ** 2) * (1.0 - p.rho ** 2) / denom


def ssvi_iv(k, T, theta, p):
    return np.sqrt(ssvi_w(k, theta, p) / T)


def ssvi_durrleman_g(k, theta, p):
    w = ssvi_w(k, theta, p)
    wp = ssvi_dw_dk(k, theta, p)
    wpp = ssvi_d2w_dk2(k, theta, p)
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


def ssvi_atm_skew(T, theta, p):
    phi = phi_power_law(theta, p.eta, p.gamma)
    return p.rho * phi * np.sqrt(theta / T) / 2.0


# ----------------------------------------------------------------
# xSSVI parameterisation (per-maturity rho)
# ----------------------------------------------------------------

def xssvi_w(k, theta, rho, eta, gamma):
    phi = eta * np.power(theta, -gamma)
    z = phi * k
    return 0.5 * theta * (1.0 + rho * z + np.sqrt((z + rho) ** 2 + 1.0 - rho ** 2))


def xssvi_dw_dk(k, theta, rho, eta, gamma):
    phi = eta * np.power(theta, -gamma)
    z = phi * k
    inner_root = np.sqrt((z + rho) ** 2 + 1.0 - rho ** 2)
    return 0.5 * theta * phi * (rho + (z + rho) / inner_root)


def xssvi_d2w_dk2(k, theta, rho, eta, gamma):
    phi = eta * np.power(theta, -gamma)
    z = phi * k
    denom = np.power((z + rho) ** 2 + 1.0 - rho ** 2, 1.5)
    return 0.5 * theta * (phi ** 2) * (1.0 - rho ** 2) / denom


def xssvi_durrleman_g(k, theta, rho, eta, gamma):
    w = xssvi_w(k, theta, rho, eta, gamma)
    wp = xssvi_dw_dk(k, theta, rho, eta, gamma)
    wpp = xssvi_d2w_dk2(k, theta, rho, eta, gamma)
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


# ----------------------------------------------------------------
# ATM theta_T term structure
# ----------------------------------------------------------------

def build_atm_theta_curve(snap, k_window=0.1):
    """ATM total variance per maturity by local quadratic. Requires
    columns: expiry, k, w, tau. Enforces monotonicity in T."""
    rows = []
    for expiry, slice_df in snap.groupby('expiry'):
        sdf = slice_df.sort_values('k').reset_index(drop=True)
        near = sdf[sdf.k.between(-k_window, k_window)]
        if len(near) >= 3:
            coeffs = np.polyfit(near.k.values, near.w.values, 2)
            theta = float(np.polyval(coeffs, 0.0))
        elif len(sdf) >= 1:
            i_nearest = int(sdf.k.abs().idxmin())
            theta = float(sdf.loc[i_nearest, 'w'])
        else:
            continue
        T = float(sdf.tau.iloc[0])
        rows.append({'expiry': expiry, 'T': T, 'theta_raw': theta, 'n_near': len(near)})

    atm = pd.DataFrame(rows).sort_values('T').reset_index(drop=True)
    atm['theta'] = np.maximum.accumulate(atm.theta_raw.values)
    atm['theta_adjusted'] = atm.theta != atm.theta_raw
    return atm


# ----------------------------------------------------------------
# SSVI calibration
# ----------------------------------------------------------------

def fit_ssvi_surface(snap, atm_curve, weights='vega',
                     rho_init=-0.6, eta_init=1.0, gamma_init=0.3,
                     verbose=False):
    """Fit SSVI to a full surface via SLSQP."""
    snap_ = snap.merge(atm_curve[['expiry', 'theta']], on='expiry', how='left')
    snap_ = snap_[snap_.theta.notna()].reset_index(drop=True)

    k_arr = snap_.k.values
    theta_arr = snap_.theta.values
    T_arr = snap_.tau.values
    iv_market = snap_.iv_mid.values

    if weights == 'vega':
        v = bs_vega(snap_.S.values, snap_.strike.values, T_arr,
                    snap_.r.values, snap_.q.values, iv_market)
        w_quote = np.where(np.isfinite(v) & (v > 1e-6), v, 0.0)
    elif weights == 'uniform':
        w_quote = np.ones_like(k_arr)
    else:
        raise ValueError(f'unknown weights option: {weights}')

    theta_min = float(atm_curve.theta.min())
    theta_max = float(atm_curve.theta.max())

    def loss(params):
        rho, eta, gamma = params
        p = SSVIParams(rho=float(rho), eta=float(eta), gamma=float(gamma))
        w_model = ssvi_w(k_arr, theta_arr, p)
        w_model = np.clip(w_model, 1e-10, None)
        iv_model = np.sqrt(w_model / T_arr)
        return float(np.sum((w_quote ** 2) * (iv_model - iv_market) ** 2))

    def butterfly_1(params, theta_val):
        rho, eta, gamma = params
        phi = eta * np.power(theta_val, -gamma)
        return 4.0 - theta_val * phi * (1.0 + abs(rho)) - 0.01

    def butterfly_2(params, theta_val):
        rho, eta, gamma = params
        phi = eta * np.power(theta_val, -gamma)
        return 4.0 - theta_val * (phi ** 2) * (1.0 + abs(rho))

    constraints = []
    for tv in (theta_min, theta_max):
        constraints.append({'type': 'ineq', 'fun': butterfly_1, 'args': (tv,)})
        constraints.append({'type': 'ineq', 'fun': butterfly_2, 'args': (tv,)})

    bounds = [(-0.999, -0.01), (0.01, 20.0), (0.01, 0.95)]

    res = minimize(loss, [rho_init, eta_init, gamma_init],
                   method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'maxiter': 300, 'disp': verbose})

    params = SSVIParams(rho=float(res.x[0]), eta=float(res.x[1]), gamma=float(res.x[2]))
    return params, res


# ----------------------------------------------------------------
# xSSVI calibration
# ----------------------------------------------------------------

def fit_xssvi_surface(snap, atm_curve, ssvi_init,
                      weights='vega',
                      n_cal_grid=25, k_cal_range=(-0.5, 0.3),
                      verbose=False):
    """Fit xSSVI: per-maturity rho, shared (eta, gamma).

    ssvi_init is an SSVIParams used as warm start. Returns dict including
    the fitted parameters and the sorted ATM curve.
    """
    sorted_atm = atm_curve.sort_values('T').reset_index(drop=True).copy()
    sorted_atm['mat_idx'] = np.arange(len(sorted_atm))
    Ts = sorted_atm['T'].values
    thetas = sorted_atm.theta.values
    N = len(Ts)

    snap_join = snap.merge(sorted_atm[['expiry', 'theta', 'mat_idx']],
                           on='expiry', how='left')
    snap_join = snap_join[snap_join.theta.notna()].reset_index(drop=True)

    quote_idx = snap_join.mat_idx.values.astype(int)
    k_arr = snap_join.k.values
    theta_arr = snap_join.theta.values
    T_arr = snap_join.tau.values
    iv_market = snap_join.iv_mid.values

    if weights == 'vega':
        v = bs_vega(snap_join.S.values, snap_join.strike.values, T_arr,
                    snap_join.r.values, snap_join.q.values, iv_market)
        w_quote = np.where(np.isfinite(v) & (v > 1e-6), v, 0.0)
    elif weights == 'uniform':
        w_quote = np.ones_like(k_arr)
    else:
        raise ValueError(f'unknown weights: {weights}')

    k_cal_grid = np.linspace(k_cal_range[0], k_cal_range[1], n_cal_grid)

    def loss(params):
        eta = params[0]; gamma = params[1]; rhos = params[2:]
        rho_per_quote = rhos[quote_idx]
        phi = eta * np.power(theta_arr, -gamma)
        z = phi * k_arr
        w_model = 0.5 * theta_arr * (1.0 + rho_per_quote * z +
                    np.sqrt((z + rho_per_quote) ** 2 + 1.0 - rho_per_quote ** 2))
        w_model = np.clip(w_model, 1e-10, None)
        iv_model = np.sqrt(w_model / T_arr)
        return float(np.sum((w_quote ** 2) * (iv_model - iv_market) ** 2))

    def butterfly_constraint_vec(params):
        eta = params[0]; gamma = params[1]; rhos = params[2:]
        phi = eta * np.power(thetas, -gamma)
        abs_rho_factor = 1.0 + np.abs(rhos)
        c1 = 4.0 - thetas * phi * abs_rho_factor - 0.01
        c2 = 4.0 - thetas * (phi ** 2) * abs_rho_factor
        return np.concatenate([c1, c2])

    def calendar_constraint_vec(params):
        eta = params[0]; gamma = params[1]; rhos = params[2:]
        THETA = thetas[:, None]
        PHI = eta * np.power(THETA, -gamma)
        RHO = rhos[:, None]
        Z = PHI * k_cal_grid[None, :]
        W = 0.5 * THETA * (1.0 + RHO * Z + np.sqrt((Z + RHO) ** 2 + 1.0 - RHO ** 2))
        diffs = W[1:, :] - W[:-1, :]
        return diffs.flatten()

    constraints = [
        {'type': 'ineq', 'fun': butterfly_constraint_vec},
        {'type': 'ineq', 'fun': calendar_constraint_vec},
    ]
    bounds = [(0.01, 20.0), (0.01, 0.95)] + [(-0.999, -0.01)] * N

    x0 = np.concatenate([
        [ssvi_init.eta, ssvi_init.gamma],
        np.full(N, ssvi_init.rho)
    ])

    res = minimize(loss, x0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'maxiter': 500, 'disp': verbose})

    return {
        'eta': float(res.x[0]),
        'gamma': float(res.x[1]),
        'rhos': res.x[2:].copy(),
        'Ts': Ts.copy(),
        'thetas': thetas.copy(),
        'expiries': list(sorted_atm.expiry),
        'sorted_atm': sorted_atm,
        'result': res,
    }


# ----------------------------------------------------------------
# Surface classes
# ----------------------------------------------------------------

class SSVISurface:
    """SSVI with ATM term structure, single rho."""

    def __init__(self, params, atm_curve, S, snapshot_date):
        self.params = params
        self.atm = atm_curve.copy()
        self.S = S
        self.snapshot_date = snapshot_date
        self._theta = interp1d(
            atm_curve['T'].values, atm_curve['theta'].values,
            kind='linear', bounds_error=False,
            fill_value=(atm_curve.theta.iloc[0], atm_curve.theta.iloc[-1]),
        )

    def theta(self, T):
        return float(self._theta(T))

    def total_variance(self, k, T):
        return ssvi_w(np.asarray(k), self.theta(T), self.params)

    def iv(self, k, T):
        return np.sqrt(self.total_variance(k, T) / T)

    def density(self, k, T):
        theta_t = self.theta(T)
        w = ssvi_w(k, theta_t, self.params)
        g = ssvi_durrleman_g(k, theta_t, self.params)
        d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
        return g * np.exp(-d2 ** 2 / 2) / np.sqrt(2 * np.pi * w)

    def atm_skew(self, T):
        return ssvi_atm_skew(T, self.theta(T), self.params)

    def __repr__(self):
        return (f'<SSVISurface date={self.snapshot_date} '
                f'rho={self.params.rho:+.3f} eta={self.params.eta:.3f} '
                f'gamma={self.params.gamma:.3f}>')


class XSSVISurface:
    """xSSVI with per-maturity rho linearly interpolated across T."""

    def __init__(self, xssvi_fit, S, snapshot_date):
        self.fit = xssvi_fit
        self.S = S
        self.snapshot_date = snapshot_date
        self.eta = xssvi_fit['eta']
        self.gamma = xssvi_fit['gamma']
        self.Ts = xssvi_fit['Ts']
        self.thetas = xssvi_fit['thetas']
        self.rhos = xssvi_fit['rhos']

        self._theta = interp1d(
            self.Ts, self.thetas, kind='linear', bounds_error=False,
            fill_value=(self.thetas[0], self.thetas[-1]),
        )
        self._rho = interp1d(
            self.Ts, self.rhos, kind='linear', bounds_error=False,
            fill_value=(self.rhos[0], self.rhos[-1]),
        )

    def theta(self, T):
        return float(self._theta(T))

    def rho(self, T):
        return float(self._rho(T))

    def total_variance(self, k, T):
        return xssvi_w(np.asarray(k), self.theta(T), self.rho(T), self.eta, self.gamma)

    def iv(self, k, T):
        return np.sqrt(self.total_variance(k, T) / T)

    def atm_skew(self, T):
        theta_t = self.theta(T)
        rho_t = self.rho(T)
        phi = phi_power_law(theta_t, self.eta, self.gamma)
        return rho_t * phi * np.sqrt(theta_t / T) / 2.0

    def __repr__(self):
        return (f'<XSSVISurface date={self.snapshot_date} '
                f'eta={self.eta:.3f} gamma={self.gamma:.3f} '
                f'rho_range=({self.rhos.min():+.3f}, {self.rhos.max():+.3f})>')
