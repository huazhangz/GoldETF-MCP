# Regenerate test/fixtures/statsmodels_reference.json (requires: pip install statsmodels)
import json, warnings
import numpy as np
warnings.simplefilter("ignore")
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
rng = np.random.default_rng(429)
cases = {}
# random walk (unit root), AR(1) stationary, trend-stationary
rw = np.cumsum(rng.normal(size=215)) + 50
ar = np.zeros(215)
for t in range(1,215): ar[t] = 0.5*ar[t-1] + rng.normal()
ts = 0.05*np.arange(215) + rng.normal(size=215)
for name, x in [("rw", rw), ("ar1", ar), ("trend", ts)]:
    c = {"x": x.tolist()}
    for reg in ["c","ct"]:
        r = adfuller(x, regression=reg, autolag="AIC")
        c[f"adf_{reg}"] = {"stat": r[0], "p": r[1], "lag": r[2], "nobs": r[3], "crit": r[4]}
        r2 = adfuller(x, regression=reg, maxlag=3, autolag=None)
        c[f"adf_{reg}_fixed3"] = {"stat": r2[0], "p": r2[1]}
        k = kpss(x, regression=reg, nlags=4)
        c[f"kpss_{reg}"] = {"stat": k[0], "p": k[1]}
    c["acf"] = acf(x, nlags=24, fft=False).tolist()
    c["pacf"] = pacf(x, nlags=24, method="ldb").tolist()
    lb = acorr_ljungbox(x, lags=24, model_df=2)
    c["lb"] = {"stat": lb["lb_stat"].tolist(), "p": [None if np.isnan(v) else v for v in lb["lb_pvalue"]]}
    cases[name] = c
import pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "test/fixtures/statsmodels_reference.json"
json.dump(cases, open(out, "w"), indent=1)
