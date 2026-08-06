"""Black-Scholes, Bjerksund-Stensland 2002, and de-Americanisation pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, linregress
from scipy.optimize import brentq

from .rates import get_rate


# ================================================================
# Black-Scholes-Merton with continuous dividend yield
# ================================================================

def bs_d1_d2(S, K, T, r, q, sigma):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S, K, T, r, q, sigma, is_call=True):
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if is_call:
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_implied_vol(price, S, K, T, r, q, is_call=True,
                   vol_lo=1e-4, vol_hi=5.0, tol=1e-8):
    disc_r = np.exp(-r * T); disc_q = np.exp(-q * T)
    if is_call:
        intrinsic = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        intrinsic = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return np.nan
    def obj(sigma):
        return float(bs_price(S, K, T, r, q, sigma, is_call) - price)
    try:
        return brentq(obj, vol_lo, vol_hi, xtol=tol, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


# ================================================================
# Bjerksund-Stensland 2002 American option pricer
# ================================================================

def _phi_bs2002(S, T, gamma, H, X, r, q, sigma):
    b = r - q
    sigma2 = sigma * sigma
    sqrtT = np.sqrt(T)
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma2
    kappa = 2.0 * b / sigma2 + 2.0 * gamma - 1.0
    d1_arg = -(np.log(S / H) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)
    d2_arg = -(np.log(X * X / (S * H)) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)
    return np.exp(lam * T) * (S ** gamma) * (norm.cdf(d1_arg) - (X / S) ** kappa * norm.cdf(d2_arg))


def bs2002_call(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    b = r - q
    if b >= r:
        return float(bs_price(S, K, T, r, q, sigma, is_call=True))
    sigma2 = sigma * sigma
    b_s2 = b / sigma2
    beta = (0.5 - b_s2) + np.sqrt((b_s2 - 0.5) ** 2 + 2.0 * r / sigma2)
    if not np.isfinite(beta) or beta > 100:
        return float(bs_price(S, K, T, r, q, sigma, is_call=True))
    B_inf = K * beta / (beta - 1.0)
    B_0 = max(K, K * r / (r - b))
    h_T = -(b * T + 2.0 * sigma * np.sqrt(T)) * (K * K) / ((B_inf - B_0) * B_0)
    X = B_0 + (B_inf - B_0) * (1.0 - np.exp(h_T))
    if S >= X:
        return S - K
    alpha = (X - K) * X ** (-beta)
    with np.errstate(over='ignore', invalid='ignore'):
        price = (alpha * S ** beta
                 - alpha * _phi_bs2002(S, T, beta, X, X, r, q, sigma)
                 + _phi_bs2002(S, T, 1.0, X, X, r, q, sigma)
                 - _phi_bs2002(S, T, 1.0, K, X, r, q, sigma)
                 - K * _phi_bs2002(S, T, 0.0, X, X, r, q, sigma)
                 + K * _phi_bs2002(S, T, 0.0, K, X, r, q, sigma))
    if not np.isfinite(price):
        return float(bs_price(S, K, T, r, q, sigma, is_call=True))
    return float(price)


def bs2002_put(S, K, T, r, q, sigma):
    return bs2002_call(K, S, T, q, r, sigma)


def bs2002_price(S, K, T, r, q, sigma, is_call=True):
    return bs2002_call(S, K, T, r, q, sigma) if is_call else bs2002_put(S, K, T, r, q, sigma)


def bs2002_implied_vol(price, S, K, T, r, q, is_call=True,
                       vol_lo=0.02, vol_hi=3.0, tol=1e-7):
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price < intrinsic - 1e-8:
        return np.nan
    def obj(sigma):
        return bs2002_price(S, K, T, r, q, sigma, is_call) - price
    try:
        lo, hi = obj(vol_lo), obj(vol_hi)
        if lo * hi > 0:
            return np.nan
        return brentq(obj, vol_lo, vol_hi, xtol=tol, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def bs2002_implied_vol_fast(price, S, K, T, r, q, sigma_init,
                             is_call=True, n_newton=8, tol=1e-7,
                             use_brent_fallback=True):
    """Newton-Raphson BS2002 IV inversion with warm start. Brent fallback."""
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price < intrinsic - 1e-8:
        return np.nan

    sigma = float(sigma_init) if (sigma_init is not None and
                                    np.isfinite(sigma_init) and
                                    0.01 < sigma_init < 3.0) else 0.20

    for _ in range(n_newton):
        f = bs2002_price(S, K, T, r, q, sigma, is_call) - price
        if abs(f) < tol:
            return sigma
        v = float(bs_vega(S, K, T, r, q, sigma))
        if v < 1e-6:
            break
        sigma_new = sigma - f / v
        if sigma_new <= 0.01 or sigma_new >= 5.0:
            break
        if abs(sigma_new - sigma) < 1e-10:
            return sigma_new
        sigma = sigma_new

    if use_brent_fallback:
        return bs2002_implied_vol(price, S, K, T, r, q, is_call=is_call)
    return np.nan


# ================================================================
# Leisen-Reimer binomial validation reference (unchanged)
# ================================================================

def leisen_reimer_price(S, K, T, r, q, sigma, n_steps=251, is_call=True, is_american=True):
    if n_steps % 2 == 0:
        n_steps += 1
    dt = T / n_steps
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    def pp2(z):
        sign = 1.0 if z >= 0 else -1.0
        return 0.5 + sign * np.sqrt(0.25 - 0.25 * np.exp(
            -((z / (n_steps + 1.0 / 3.0)) ** 2) * (n_steps + 1.0 / 6.0)))
    p_prime, p = pp2(d1), pp2(d2)
    u = np.exp((r - q) * dt) * p_prime / p
    d = (np.exp((r - q) * dt) - p * u) / (1 - p)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    S_T = S * (u ** j) * (d ** (n_steps - j))
    V = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if is_american:
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            payoff = (S_i - K) if is_call else (K - S_i)
            V = np.maximum(V, payoff)
    return float(V[0])


# ================================================================
# De-Americanisation pipeline (Brent baseline)
# ================================================================

def deamericanise_slice(strikes, c_mid, p_mid, S, T, r,
                        q_init=0.015, q_tol=0.0005, max_iter=4,
                        q_cap=0.10, diag=None):
    strikes = np.asarray(strikes, dtype=float)
    c_mid = np.asarray(c_mid, dtype=float)
    p_mid = np.asarray(p_mid, dtype=float)
    q = q_init
    q_implied = q_init

    for iteration in range(1, max_iter + 1):
        iv_call = np.array([bs2002_implied_vol(c, S, K, T, r, q, True)
                             for K, c in zip(strikes, c_mid)])
        iv_put = np.array([bs2002_implied_vol(p, S, K, T, r, q, False)
                            for K, p in zip(strikes, p_mid)])
        valid = ~(np.isnan(iv_call) | np.isnan(iv_put))
        if diag is not None:
            diag['n_invert_valid'] = int(valid.sum())
        if valid.sum() < 5:
            if diag is not None:
                diag['fail_reason'] = f'only {valid.sum()} strikes inverted'
            return None

        K_v = strikes[valid]
        C_E = bs_price(S, K_v, T, r, q, iv_call[valid], is_call=True)
        P_E = bs_price(S, K_v, T, r, q, iv_put[valid], is_call=False)
        slope, intercept, *_ = linregress(K_v, C_E - P_E)
        D = -slope
        if D <= 0 or D > 1.5:
            if diag is not None: diag['fail_reason'] = f'bad D: {D:.4f}'
            return None
        F = intercept / D
        if F < 0.3 * S or F > 3.0 * S:
            if diag is not None: diag['fail_reason'] = f'bad F: {F:.2f}'
            return None
        q_implied = -np.log(F / (S * np.exp(r * T))) / T
        q_new = float(np.clip(q_implied, 0.0, q_cap))
        if abs(q_new - q) < q_tol:
            q = q_new
            break
        q = q_new

    k = np.log(strikes / F)
    iv_mid = np.where(k >= 0, iv_call, iv_put)
    return {
        'strike': strikes, 'k': k,
        'iv_call': iv_call, 'iv_put': iv_put, 'iv_mid': iv_mid,
        'F': F, 'q': q, 'q_implied': q_implied, 'D': D,
        'n_iter': iteration, 'n_valid': int(valid.sum()),
    }


def deamericanise_snapshot(date, df, treasury_df, min_strikes=10, verbose=False):
    day = df[df.QUOTE_DATE == pd.Timestamp(date)].copy()
    day = day[(day.C_BID > 0) & (day.C_ASK > day.C_BID) &
              (day.P_BID > 0) & (day.P_ASK > day.P_BID)]
    day['C_MID'] = (day.C_BID + day.C_ASK) / 2
    day['P_MID'] = (day.P_BID + day.P_ASK) / 2
    S = day.UNDERLYING_LAST.iloc[0]
    day = day[day.STRIKE.between(0.5 * S, 1.5 * S)]

    out_rows, diagnostics = [], []
    for expiry, slice_df in day.groupby('EXPIRE_DATE'):
        diag = {'expiry': expiry, 'T': slice_df.T_years.iloc[0],
                'n_strikes_filtered': len(slice_df), 'fail_reason': None}
        if len(slice_df) < min_strikes:
            diag['fail_reason'] = f'only {len(slice_df)} strikes'
            diagnostics.append(diag); continue
        T = slice_df.T_years.iloc[0]
        r = get_rate(date, T, treasury_df)
        if not np.isfinite(r):
            diag['fail_reason'] = 'no rate'; diagnostics.append(diag); continue

        result = deamericanise_slice(
            slice_df.STRIKE.values, slice_df.C_MID.values, slice_df.P_MID.values,
            S=S, T=T, r=r, diag=diag,
        )
        diagnostics.append(diag)
        if result is None: continue
        out_rows.append(pd.DataFrame({
            'date': pd.Timestamp(date), 'expiry': expiry, 'T': T,
            'r': r, 'F': result['F'], 'q': result['q'],
            'q_implied': result['q_implied'], 'D': result['D'],
            'n_iter': result['n_iter'], 'n_valid': result['n_valid'],
            'strike': result['strike'], 'k': result['k'],
            'iv_call': result['iv_call'], 'iv_put': result['iv_put'],
            'iv_mid': result['iv_mid'], 'S': S,
        }))

    diag_df = pd.DataFrame(diagnostics)
    if verbose:
        print(diag_df.to_string(index=False))
    if not out_rows:
        return pd.DataFrame(), diag_df
    return pd.concat(out_rows, ignore_index=True), diag_df


# ================================================================
# De-Americanisation pipeline (Newton fast path with warm starts)
# ================================================================

def deamericanise_slice_fast(strikes, c_mid, p_mid, c_iv_seed, p_iv_seed,
                              S, T, r, q_init=0.015, q_tol=0.0005,
                              max_iter=4, q_cap=0.10, diag=None):
    """Newton-warm-start de-Am of one expiry slice. ~10x faster than Brent.

    c_iv_seed and p_iv_seed are warm-start IVs (e.g. OptionsDX precomputed
    C_IV and P_IV); on NaN/garbage values the inverter falls back to 0.20.
    """
    strikes = np.asarray(strikes, dtype=float)
    c_mid = np.asarray(c_mid, dtype=float)
    p_mid = np.asarray(p_mid, dtype=float)
    c_iv_seed = np.asarray(c_iv_seed, dtype=float)
    p_iv_seed = np.asarray(p_iv_seed, dtype=float)
    q = q_init
    q_implied = q_init

    for iteration in range(1, max_iter + 1):
        iv_call = np.array([
            bs2002_implied_vol_fast(c, S, K, T, r, q, seed, is_call=True)
            for K, c, seed in zip(strikes, c_mid, c_iv_seed)
        ])
        iv_put = np.array([
            bs2002_implied_vol_fast(p, S, K, T, r, q, seed, is_call=False)
            for K, p, seed in zip(strikes, p_mid, p_iv_seed)
        ])
        valid = ~(np.isnan(iv_call) | np.isnan(iv_put))
        if diag is not None:
            diag['n_invert_valid'] = int(valid.sum())
        if valid.sum() < 5:
            if diag is not None:
                diag['fail_reason'] = f'only {valid.sum()} inversions'
            return None

        K_v = strikes[valid]
        C_E = bs_price(S, K_v, T, r, q, iv_call[valid], is_call=True)
        P_E = bs_price(S, K_v, T, r, q, iv_put[valid], is_call=False)
        slope, intercept, *_ = linregress(K_v, C_E - P_E)
        D = -slope
        if D <= 0 or D > 1.5:
            if diag is not None: diag['fail_reason'] = f'bad D: {D:.4f}'
            return None
        F = intercept / D
        if F < 0.3 * S or F > 3.0 * S:
            if diag is not None: diag['fail_reason'] = f'bad F: {F:.2f}'
            return None
        q_implied = -np.log(F / (S * np.exp(r * T))) / T
        q_new = float(np.clip(q_implied, 0.0, q_cap))
        if abs(q_new - q) < q_tol:
            q = q_new
            break
        q = q_new

    k = np.log(strikes / F)
    iv_mid = np.where(k >= 0, iv_call, iv_put)
    return {
        'strike': strikes, 'k': k,
        'iv_call': iv_call, 'iv_put': iv_put, 'iv_mid': iv_mid,
        'F': F, 'q': q, 'q_implied': q_implied, 'D': D,
        'n_iter': iteration, 'n_valid': int(valid.sum()),
    }


def deamericanise_snapshot_fast(date, df, treasury_df, min_strikes=10, verbose=False):
    """Newton-warm-start de-Am for a full trading day.

    Requires df to have OptionsDX C_IV and P_IV columns for warm starts.
    """
    day = df[df.QUOTE_DATE == pd.Timestamp(date)].copy()
    day = day[(day.C_BID > 0) & (day.C_ASK > day.C_BID) &
              (day.P_BID > 0) & (day.P_ASK > day.P_BID)]
    day['C_MID'] = (day.C_BID + day.C_ASK) / 2
    day['P_MID'] = (day.P_BID + day.P_ASK) / 2
    S = day.UNDERLYING_LAST.iloc[0]
    day = day[day.STRIKE.between(0.5 * S, 1.5 * S)]

    out_rows, diagnostics = [], []
    for expiry, slice_df in day.groupby('EXPIRE_DATE'):
        diag = {'expiry': expiry, 'T': slice_df.T_years.iloc[0],
                'n_strikes_filtered': len(slice_df), 'fail_reason': None}
        if len(slice_df) < min_strikes:
            diag['fail_reason'] = f'only {len(slice_df)} strikes'
            diagnostics.append(diag); continue
        T = slice_df.T_years.iloc[0]
        r = get_rate(date, T, treasury_df)
        if not np.isfinite(r):
            diag['fail_reason'] = 'no rate'; diagnostics.append(diag); continue

        result = deamericanise_slice_fast(
            slice_df.STRIKE.values, slice_df.C_MID.values, slice_df.P_MID.values,
            slice_df.C_IV.values, slice_df.P_IV.values,
            S=S, T=T, r=r, diag=diag,
        )
        diagnostics.append(diag)
        if result is None: continue
        out_rows.append(pd.DataFrame({
            'date': pd.Timestamp(date), 'expiry': expiry, 'T': T,
            'r': r, 'F': result['F'], 'q': result['q'],
            'q_implied': result['q_implied'], 'D': result['D'],
            'n_iter': result['n_iter'], 'n_valid': result['n_valid'],
            'strike': result['strike'], 'k': result['k'],
            'iv_call': result['iv_call'], 'iv_put': result['iv_put'],
            'iv_mid': result['iv_mid'], 'S': S,
        }))

    diag_df = pd.DataFrame(diagnostics)
    if verbose:
        print(diag_df.to_string(index=False))
    if not out_rows:
        return pd.DataFrame(), diag_df
    return pd.concat(out_rows, ignore_index=True), diag_df
