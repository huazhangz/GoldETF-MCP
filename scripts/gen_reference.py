# Regenerate the reference fixtures in test/fixtures/ from statsmodels, arch and scipy.
#   python -m venv .venv && .venv/bin/pip install statsmodels arch
#   .venv/bin/python scripts/gen_reference.py
# All inputs are synthetic (seeded), so the fixtures can be committed.
import json
import math
import pathlib
import warnings

import numpy as np
from scipy import stats

warnings.simplefilter("ignore")

from arch import arch_model  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402
from statsmodels.stats.stattools import jarque_bera  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "test/fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    def clean(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return clean(o.item())
        return o

    (OUT / name).write_text(json.dumps(clean(obj), indent=1))


# --------------------------------------------------------------------------
# 1. Stationarity / autocorrelation (statsmodels_reference.json)
# --------------------------------------------------------------------------
rng = np.random.default_rng(429)
cases = {}
rw = np.cumsum(rng.normal(size=215)) + 50
ar = np.zeros(215)
for t in range(1, 215):
    ar[t] = 0.5 * ar[t - 1] + rng.normal()
ts = 0.05 * np.arange(215) + rng.normal(size=215)
for name, x in [("rw", rw), ("ar1", ar), ("trend", ts)]:
    c = {"x": x.tolist()}
    for reg in ["c", "ct"]:
        r = adfuller(x, regression=reg, autolag="AIC")
        c[f"adf_{reg}"] = {"stat": r[0], "p": r[1], "lag": r[2], "nobs": r[3], "crit": r[4]}
        r2 = adfuller(x, regression=reg, maxlag=3, autolag=None)
        c[f"adf_{reg}_fixed3"] = {"stat": r2[0], "p": r2[1]}
        k = kpss(x, regression=reg, nlags=4)
        c[f"kpss_{reg}"] = {"stat": k[0], "p": k[1]}
    c["acf"] = acf(x, nlags=24, fft=False).tolist()
    c["pacf"] = pacf(x, nlags=24, method="ldb").tolist()
    lb = acorr_ljungbox(x, lags=24, model_df=2)
    c["lb"] = {"stat": lb["lb_stat"].tolist(), "p": lb["lb_pvalue"].tolist()}
    cases[name] = c
dump("statsmodels_reference.json", cases)

# --------------------------------------------------------------------------
# 2. ARIMA (arima_reference.json)
# --------------------------------------------------------------------------
rng = np.random.default_rng(2026)
n = 215
e = rng.standard_t(5, size=n + 1) * 0.12
w = np.zeros(n)
for t in range(1, n):
    w[t] = 0.85 * w[t - 1] + e[t] + 0.2 * e[t - 1]
level = 2.9 + w  # mean-reverting log-vol-like series
dd = np.zeros(n)
for t in range(1, n):
    dd[t] = 0.4 * dd[t - 1] + e[t] - 0.3 * e[t - 1]
integrated = 3.0 + np.cumsum(dd + 0.002)
arima_cases = []
for y, order in [
    (integrated, (0, 1, 0)),
    (integrated, (1, 1, 1)),
    (integrated, (2, 1, 2)),
    (level, (1, 0, 0)),
    (level, (2, 0, 1)),
    (level, (0, 0, 2)),
]:
    f = ARIMA(y, order=order, trend="c" if order[1] == 0 else "t").fit()
    fh = ARIMA(y, order=order, trend="c" if order[1] == 0 else "t").fit(cov_type="approx")
    fc = f.get_forecast(8)
    arima_cases.append(
        {
            "y": y.tolist(),
            "order": order,
            "params": f.params.tolist(),
            "llf": f.llf,
            "aic": f.aic,
            "aicc": f.aicc,
            "bic": f.bic,
            "bse_hessian": fh.bse.tolist(),
            "fmean": fc.predicted_mean.tolist(),
            "fvar": fc.var_pred_mean.tolist(),
            "resid": f.resid[order[1]:].tolist(),
        }
    )
dump("arima_reference.json", arima_cases)

# --------------------------------------------------------------------------
# 3. Volatility models (arch_reference.json)
# --------------------------------------------------------------------------
rng = np.random.default_rng(7)
n = 1500
nu = 6.0
z = rng.standard_t(nu, size=n) / math.sqrt(nu / (nu - 2))
y = np.zeros(n)
s2 = 0.05 / (1 - 0.95)
eps_prev = 0.0
for t in range(n):
    s2 = 0.05 + 0.1 * eps_prev**2 + 0.85 * s2 if t > 0 else s2
    eps = math.sqrt(s2) * z[t]
    y[t] = 0.1 + (0.3 * y[t - 1] if t > 0 else 0) + eps
    eps_prev = eps
vol_cases = []
for kw in [
    dict(mean="Constant", vol="GARCH", dist="normal"),
    dict(mean="Constant", vol="GARCH", dist="t"),
    dict(mean="AR", lags=1, vol="GARCH", dist="t"),
    dict(mean="AR", lags=1, vol="Constant", dist="t"),
    dict(mean="Zero", vol="GARCH", dist="normal"),
]:
    r = arch_model(y, rescale=False, **kw).fit(disp="off")
    fc = r.forecast(horizon=5, reindex=False)
    vol_cases.append(
        {
            "kw": kw,
            "names": list(r.params.index),
            "params": r.params.tolist(),
            "llf": r.loglikelihood,
            "se": r.std_err.tolist(),
            "fmean": fc.mean.values[0].tolist(),
            "fvar": fc.variance.values[0].tolist(),
            "rvar": fc.residual_variance.values[0].tolist(),
        }
    )
dump("arch_reference.json", {"y": y.tolist(), "cases": vol_cases})

# --------------------------------------------------------------------------
# 4. STL, ARCH-LM, Jarque-Bera, distributions (misc_reference.json)
# --------------------------------------------------------------------------
rng = np.random.default_rng(12)
n = 180
t = np.arange(n)
x = 3 + 0.004 * t + 0.3 * np.sin(2 * np.pi * t / 70) + 0.08 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, n)
x[[40, 41, 130]] += [0.9, 0.5, -0.7]  # crisis-like outliers
ns = 10 * n + 1
nt = math.ceil(1.5 * 12 / (1 - 1.5 / ns))
nt = nt + 1 if nt % 2 == 0 else nt
stl_out = {"x": x.tolist()}
for key, kwargs, fit_kw, periodic in [
    ("periodic_robust", dict(seasonal=ns, seasonal_deg=0, trend=nt, low_pass=13, robust=True), dict(inner_iter=1, outer_iter=15), True),
    ("periodic_plain", dict(seasonal=ns, seasonal_deg=0, trend=nt, low_pass=13, robust=False), dict(inner_iter=2, outer_iter=0), True),
    ("s13_robust", dict(seasonal=13, robust=True), dict(inner_iter=1, outer_iter=15), False),
]:
    r = STL(x, period=12, **kwargs).fit(**fit_kw)
    s = np.asarray(r.seasonal)
    if periodic:
        s = np.array([s[i::12].mean() for i in range(12)])[np.arange(n) % 12]
    stl_out[key] = {"seasonal": s.tolist(), "trend": list(r.trend), "weights": list(r.weights)}

resid = np.diff(x)
lm = het_arch(resid - resid.mean(), nlags=6)
jb = jarque_bera(resid)
dist = {
    "normal_q": [[p, stats.norm.ppf(p)] for p in [1e-6, 0.001, 0.025, 0.1, 0.5, 0.8, 0.975, 0.999999]],
    "t_cdf": [[x_, df, stats.t.cdf(x_, df)] for x_, df in [(-3.2, 3), (-1.0, 4.3), (0.4, 10), (2.1, 141), (5.0, 2.5)]],
    "t_q": [[p, df, stats.t.ppf(p, df)] for p, df in [(0.9, 4.3), (0.975, 3), (0.025, 7.5), (0.995, 30)]],
}
dump("misc_reference.json", {"stl": stl_out, "archlm": {"lm": lm[0], "p": lm[1]}, "jb": list(jb), "dist": dist})
print("fixtures written to", OUT)
