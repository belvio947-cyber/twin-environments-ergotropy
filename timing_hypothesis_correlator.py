#!/usr/bin/env python3
"""
Timing hypothesis measurement-program simulation (classical correlator).
Adversarial: defines the signature that would discriminate H0 vs H1.
Does NOT claim quantum advantage was observed.

H0: independent classical jitter per channel
H1 candidate: shared latent + local noise (latent strength tied to S~1.3 bits)
Negative control: classical common-mode noise (same correlation, not quantum)

Author: measurement-program appendix for Paper 2
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N_SHOTS = 50_000  # Monte Carlo shots per point

# --- Entropy calibration ---
# For a two-outcome correlated latent with correlation rho in [-1,1],
# a simple binary model gives S related to correlation strength.
# We use: fraction of variance that is shared, f_corr in [0,1].
# Schmidt-effective rank ~6 and S~1.3 bits motivate moderate f_corr, not 1.
# We SCAN f_corr rather than fix one value.

def classical_H0(sigma_cls, n=N_SHOTS):
    """Independent Gaussian jitter on two channels from common t0."""
    t0 = 0.0
    e1 = rng.normal(0.0, sigma_cls, n)
    e2 = rng.normal(0.0, sigma_cls, n)
    t1, t2 = t0 + e1, t0 + e2
    return t1, t2

def twin_H1(sigma_local, sigma_shared, sigma_switch, n=N_SHOTS):
    """
    Minimal readout model:
      q_i = shared + local_i   (continuous proxy for module observable)
      edge from threshold crossing approximated as time shift ~ -gain * q_i
      plus independent switch/ADC jitter.
    Shared component emulates correlation from entanglement (classical latent).
    """
    shared = rng.normal(0.0, sigma_shared, n)
    loc1 = rng.normal(0.0, sigma_local, n)
    loc2 = rng.normal(0.0, sigma_local, n)
    # threshold-to-time map: larger q -> earlier edge (arbitrary linear gain=1 in time units)
    t1 = -(shared + loc1) + rng.normal(0.0, sigma_switch, n)
    t2 = -(shared + loc2) + rng.normal(0.0, sigma_switch, n)
    return t1, t2

def classical_common(sigma_local, sigma_common, sigma_switch, n=N_SHOTS):
    """Negative control: same structure as H1 but labeled classical common noise."""
    return twin_H1(sigma_local, sigma_common, sigma_switch, n)

def metrics(t1, t2):
    d = t1 - t2
    sig = float(np.std(d, ddof=1))
    c = float(np.corrcoef(t1, t2)[0, 1])
    return sig, c

def f_corr_to_sigmas(sigma_total, f_corr):
    """
    Partition total readout variance: Var = sigma_shared^2 + sigma_local^2
    f_corr = sigma_shared^2 / Var
    """
    f_corr = np.clip(f_corr, 0.0, 1.0)
    var = sigma_total ** 2
    sig_s = np.sqrt(f_corr * var)
    sig_l = np.sqrt((1.0 - f_corr) * var)
    return sig_l, sig_s

# ========================= Phase 1: H0 baseline =========================
print("=" * 60)
print("PHASE 1 — Classical baseline H0")
print("=" * 60)
sigma_cls_grid = [0.05, 0.10, 0.20, 0.50]  # arbitrary time units
print(f"{'sigma_cls':>10} {'sigma_ij':>10} {'sqrt2*sig':>10} {'C_ij':>10}")
for sc in sigma_cls_grid:
    t1, t2 = classical_H0(sc)
    sij, cij = metrics(t1, t2)
    print(f"{sc:10.3f} {sij:10.4f} {np.sqrt(2)*sc:10.4f} {cij:10.4f}")

# ========================= Phase 2+3: H1 sweep =========================
print("\n" + "=" * 60)
print("PHASE 2–3 — H1 candidate vs H0  (Delta_sigma = sigma_H0 - sigma_H1)")
print("=" * 60)

# Fix classical single-channel jitter reference so H0 is comparable
SIGMA_CLS = 0.20
t1, t2 = classical_H0(SIGMA_CLS)
sigma_H0, C_H0 = metrics(t1, t2)
print(f"H0 reference: sigma_cls={SIGMA_CLS}, sigma_ij={sigma_H0:.4f}, C={C_H0:.4f}")

# Readout total noise scale matched to classical edge noise order of magnitude
SIGMA_RO = 0.20
f_grid = np.linspace(0.0, 1.0, 11)
switch_grid = [0.0, 0.05, 0.10, 0.20, 0.50]

# results[f_idx, s_idx] = delta_sigma
delta = np.zeros((len(f_grid), len(switch_grid)))
C_h1 = np.zeros_like(delta)
sig_h1 = np.zeros_like(delta)

print(f"\n{'f_corr':>8} {'sig_sw':>8} {'sig_H1':>10} {'Delta_sig':>10} {'C_H1':>10} {'adv?':>6}")
for i, f in enumerate(f_grid):
    sig_l, sig_s = f_corr_to_sigmas(SIGMA_RO, f)
    for j, ssw in enumerate(switch_grid):
        t1, t2 = twin_H1(sig_l, sig_s, ssw)
        sij, cij = metrics(t1, t2)
        dsig = sigma_H0 - sij
        delta[i, j] = dsig
        C_h1[i, j] = cij
        sig_h1[i, j] = sij
        flag = "YES" if dsig > 0 else "no"
        if abs(f * 10 - round(f * 10)) < 1e-9 and ssw in (0.0, 0.10, 0.20, 0.50):
            print(f"{f:8.2f} {ssw:8.2f} {sij:10.4f} {dsig:10.4f} {cij:10.4f} {flag:>6}")

# Region where Delta_sigma > 0
pos = np.argwhere(delta > 0)
print(f"\nPoints with Delta_sigma > 0: {len(pos)} / {delta.size}")
if len(pos):
    print("  (f_corr, sigma_switch) examples:")
    for (i, j) in pos[:15]:
        print(f"    f={f_grid[i]:.2f}, sw={switch_grid[j]:.2f}, "
              f"Delta={delta[i,j]:.4f}, C={C_h1[i,j]:.4f}")

# ========================= Phase 4: negative control =========================
print("\n" + "=" * 60)
print("PHASE 4 — Negative control (classical common noise)")
print("=" * 60)
print("Same (f_corr, switch) as H1 structure — labels only differ.")
print(f"{'f_corr':>8} {'sig_sw':>8} {'Delta_cls':>10} {'Delta_H1':>10} {'same?':>8}")
same_count = 0
for i, f in enumerate([0.0, 0.3, 0.5, 0.7, 1.0]):
    sig_l, sig_s = f_corr_to_sigmas(SIGMA_RO, f)
    for ssw in [0.0, 0.10, 0.20]:
        t1c, t2c = classical_common(sig_l, sig_s, ssw)
        t1h, t2h = twin_H1(sig_l, sig_s, ssw)
        dc = sigma_H0 - metrics(t1c, t2c)[0]
        dh = sigma_H0 - metrics(t1h, t2h)[0]
        same = abs(dc - dh) < 0.01
        same_count += int(same)
        print(f"{f:8.2f} {ssw:8.2f} {dc:10.4f} {dh:10.4f} {'YES' if same else 'no':>8}")

print("\nCONCLUSION (adversarial):")
print("  Any Delta_sigma > 0 arises from SHARED VARIANCE, not from a quantum label.")
print("  Classical common-mode noise reproduces the same Delta_sigma.")
print("  Therefore this correlator does NOT establish a specifically quantum advantage.")
print("  A real bench must compare twin path vs classical common-mode control, not only vs H0.")
print("DONE")

# ========================= Figures =========================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=140)

# Heatmap Delta_sigma
im = axes[0].imshow(delta.T, origin="lower", aspect="auto",
                    extent=[f_grid[0], f_grid[-1], switch_grid[0], switch_grid[-1]],
                    cmap="RdBu_r", vmin=-0.3, vmax=0.3)
axes[0].set_xlabel(r"shared variance fraction $f_{\mathrm{corr}}$")
axes[0].set_ylabel(r"switch jitter $\sigma_{\mathrm{sw}}$")
axes[0].set_title(r"$\Delta\sigma = \sigma_{H_0}-\sigma_{H_1}$")
plt.colorbar(im, ax=axes[0], fraction=0.046)

# Line cuts at fixed switch noise
for ssw, style in [(0.0, "-"), (0.10, "--"), (0.20, "-."), (0.50, ":")]:
    j = switch_grid.index(ssw)
    axes[1].plot(f_grid, delta[:, j], style, label=fr"$\sigma_{{sw}}={ssw}$")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel(r"$f_{\mathrm{corr}}$")
axes[1].set_ylabel(r"$\Delta\sigma$")
axes[1].set_title("Advantage region only if $\Delta\sigma>0$")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("/home/workdir/artifacts/timing_delta_sigma.png", bbox_inches="tight")
plt.close()

# Correlation vs f_corr
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=140)
j0 = switch_grid.index(0.10)
ax.plot(f_grid, C_h1[:, j0], "o-", label=r"$C_{ij}$ (H1, $\sigma_{sw}=0.1$)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"$f_{\mathrm{corr}}$")
ax.set_ylabel(r"$C_{ij}$")
ax.set_title("Channel correlation grows with shared variance")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("/home/workdir/artifacts/timing_correlation.png", bbox_inches="tight")
plt.close()
print("Figures saved.")
