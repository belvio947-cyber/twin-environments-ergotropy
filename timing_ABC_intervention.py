#!/usr/bin/env python3
"""
Measurement program: Controls A/B/C + coupling intervention + power analysis.
Path Twin = TFIM modules + frontier-spin link (numerical only).
sigma = std(t_i - t_j) of digital edges from thresholded readout proxies.

Does NOT claim quantum advantage.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(2026)
N = 80_000  # shots per condition

def edges_independent(sigma_cls, n=N):
    """Control A: independent classical jitter."""
    e1 = rng.normal(0, sigma_cls, n)
    e2 = rng.normal(0, sigma_cls, n)
    return e1, e2

def edges_shared(sigma_local, sigma_shared, sigma_sw, n=N):
    """Shared latent + local + switch (used for twin proxy and classical B)."""
    shared = rng.normal(0, sigma_shared, n)
    t1 = -(shared + rng.normal(0, sigma_local, n)) + rng.normal(0, sigma_sw, n)
    t2 = -(shared + rng.normal(0, sigma_local, n)) + rng.normal(0, sigma_sw, n)
    return t1, t2

def f_to_sigmas(sigma_ro, f):
    f = float(np.clip(f, 0, 1))
    return np.sqrt((1 - f) * sigma_ro**2), np.sqrt(f * sigma_ro**2)

def summary(t1, t2, name=""):
    d = t1 - t2
    out = {
        "sigma": float(np.std(d, ddof=1)),
        "mean_d": float(np.mean(d)),
        "C": float(np.corrcoef(t1, t2)[0, 1]),
        "skew_d": float(stats.skew(d)),
        "kurt_d": float(stats.kurtosis(d)),
        "q05": float(np.quantile(d, 0.05)),
        "q95": float(np.quantile(d, 0.95)),
        "std_t1": float(np.std(t1, ddof=1)),
        "std_t2": float(np.std(t2, ddof=1)),
    }
    if name:
        print(f"  {name}: sigma={out['sigma']:.4f}  C={out['C']:.4f}  "
              f"skew={out['skew_d']:.3f}  kurt={out['kurt_d']:.3f}  "
              f"q05={out['q05']:.4f}  q95={out['q95']:.4f}")
    return out

def wasserstein_delta(t1a, t2a, t1b, t2b):
    """1D Wasserstein between Delta-t distributions."""
    return float(stats.wasserstein_distance(t1a - t2a, t1b - t2b))

# ---------------------------------------------------------------------------
# Operating point (matched noise scales)
# ---------------------------------------------------------------------------
SIGMA_CLS = 0.20
SIGMA_RO = 0.20
SIGMA_SW = 0.05
# Twin at nominal coupling: moderate shared fraction (inspired by S~1.3, rank~6)
F_NOMINAL = 0.45

print("=" * 64)
print("STEP 1 — Calibrate A and B against twin (C) at nominal coupling")
print("=" * 64)

sig_l, sig_s = f_to_sigmas(SIGMA_RO, F_NOMINAL)
t1_C, t2_C = edges_shared(sig_l, sig_s, SIGMA_SW)
sC = summary(t1_C, t2_C, "C  twin proxy")

t1_A, t2_A = edges_independent(SIGMA_CLS)
sA = summary(t1_A, t2_A, "A  independent")

# Control B: same generative structure as twin proxy (classical common noise)
t1_B, t2_B = edges_shared(sig_l, sig_s, SIGMA_SW)
sB = summary(t1_B, t2_B, "B  classical common")

print(f"\n  Delta_ind = sigma_A - sigma_C = {sA['sigma'] - sC['sigma']:+.4f}")
print(f"  D         = sigma_B - sigma_C = {sB['sigma'] - sC['sigma']:+.4f}")
print(f"  Wasserstein(B,C) on Delta-t   = {wasserstein_delta(t1_B,t2_B,t1_C,t2_C):.5f}")
print(f"  |C_B - C_C|                   = {abs(sB['C']-sC['C']):.5f}")
print("  -> With identical generators, B matches C: D~0 (expected).")

# ---------------------------------------------------------------------------
# STEP 2 — Intervention: map alpha_f -> f_corr for "quantum" path only
# Classical B holds f fixed at nominal (shared noise not tied to alpha_f)
# Quantum path: f_corr(alpha) = F_NOMINAL * (alpha / alpha0)^2  clipped
# (motivated by weak coupling expansion of interaction strength)
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("STEP 2 — Intervention on coupling alpha_f")
print("  Quantum path: f_corr scales with alpha_f")
print("  Classical B:  f_corr fixed at nominal (noise sources held fixed)")
print("=" * 64)

ALPHA0 = 1.5
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

print(f"{'alpha':>7} {'f_q':>6} {'sig_C':>8} {'sig_B':>8} {'D':>9} "
      f"{'C_C':>7} {'C_B':>7} {'pred_diff?':>10}")

rows = []
for a in alphas:
    f_q = F_NOMINAL * min(1.0, (a / ALPHA0) ** 2)
    sl, ss = f_to_sigmas(SIGMA_RO, f_q)
    t1c, t2c = edges_shared(sl, ss, SIGMA_SW)
    # B: fixed nominal shared noise (classical sources unchanged by alpha)
    slb, ssb = f_to_sigmas(SIGMA_RO, F_NOMINAL)
    t1b, t2b = edges_shared(slb, ssb, SIGMA_SW)
    sc = summary(t1c, t2c)
    sb = summary(t1b, t2b)
    D = sb["sigma"] - sc["sigma"]
    # Models predict differently when alpha != ALPHA0
    pred = "YES" if abs(a - ALPHA0) > 1e-9 else "same"
    print(f"{a:7.2f} {f_q:6.3f} {sc['sigma']:8.4f} {sb['sigma']:8.4f} {D:9.4f} "
          f"{sc['C']:7.3f} {sb['C']:7.3f} {pred:>10}")
    rows.append((a, f_q, sc["sigma"], sb["sigma"], D, sc["C"], sb["C"]))

print("\n  Interpretation:")
print("  At alpha=0, quantum path loses shared variance -> sigma_C rises toward A.")
print("  Classical B stays at nominal correlation -> D becomes negative (B tighter).")
print("  That differential response is the discriminant the lab should look for.")
print("  (Still a model assumption: real classical noise might also depend on alpha.)")

# ---------------------------------------------------------------------------
# STEP 3 — Statistical power for detecting D
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("STEP 3 — Statistical power (bootstrap)")
print("=" * 64)

def bootstrap_sigma(t1, t2, n_boot=400, n_use=None):
    n = len(t1) if n_use is None else n_use
    sigs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(t1), n)
        sigs.append(np.std(t1[idx] - t2[idx], ddof=1))
    return np.array(sigs)

# Effect size example: alpha=0 vs B nominal
sl0, ss0 = f_to_sigmas(SIGMA_RO, 0.0)
t1_a0, t2_a0 = edges_shared(sl0, ss0, SIGMA_SW)
t1_bn, t2_bn = edges_shared(*f_to_sigmas(SIGMA_RO, F_NOMINAL), SIGMA_SW)
true_D = np.std(t1_bn - t2_bn, ddof=1) - np.std(t1_a0 - t2_a0, ddof=1)
print(f"  Example true D (B nominal - C at alpha=0): {true_D:+.4f}")
print(f"  Relative to sigma_A~{sA['sigma']:.3f}: {100*abs(true_D)/sA['sigma']:.1f}%")

for n_shots in [500, 1000, 2000, 5000, 10000, 20000]:
    bB = bootstrap_sigma(t1_bn, t2_bn, n_use=n_shots)
    bC = bootstrap_sigma(t1_a0, t2_a0, n_use=n_shots)
    Dboot = bB - bC
    # two-sided: fraction of bootstrap D with same sign as true_D and |D|>0
    # power proxy: CI excludes 0
    lo, hi = np.quantile(Dboot, [0.025, 0.975])
    excludes0 = (lo > 0 and true_D > 0) or (hi < 0 and true_D < 0)
    print(f"  N={n_shots:5d}  mean_D={Dboot.mean():+.4f}  "
          f"95% CI=[{lo:+.4f},{hi:+.4f}]  excludes0={excludes0}")

print("\n" + "=" * 64)
print("VERDICT (adversarial)")
print("=" * 64)
print("1. Matched generators: control B reproduces twin C (D~0, Wasserstein~0).")
print("2. Intervention alpha_f->0: quantum path and fixed classical B diverge:")
print("   D changes sign/magnitude — this is a usable discriminant *if*")
print("   laboratory classical noise truly stays fixed when alpha is changed.")
print("3. For ~10% effects, a few thousand shots often exclude D=0 in this model;")
print("   real noise will demand more. Thresholds are design targets, not claims.")
print("4. No quantum advantage is asserted. Program of measurement only.")
print("DONE")
