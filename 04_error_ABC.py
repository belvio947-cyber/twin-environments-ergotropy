#!/usr/bin/env python3
"""Pegar TODO en UNA celda de Kaggle y Run. Sin argparse."""
#!/usr/bin/env python3
# ============================================================================
# TEBD full-chain — twin environments N=2..10
# Optimizado para Kaggle (complex64 en extension, gates precomputados,
# S(mid) por SVD del bond — NO contraccion densa D^(N/2)).
#
# - TODOS los links NN activos durante t_link
# - Mide E_res y S desde el MPS (no fija E = N*E_M)
# - Checksum N=2,3 obligatorio antes de subir N
# - TEBD/Trotter aproximado, no ED exacta
#
# Uso:
#   python tebd_kaggle_N2_to_N10.py
#   python tebd_kaggle_N2_to_N10.py --max-n 8
#   python tebd_kaggle_N2_to_N10.py --checksum-only
# ============================================================================
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from typing import List, Optional

import numpy as np
from numpy.linalg import norm, svd
from scipy.linalg import expm

# -------------------- parametros (papers) --------------------
J, HFIELD = 1.0, 2.0
NQ, NF = 4, 2
D_SPIN = 2**NQ  # 16
D = D_SPIN * NF  # 32
OMEGA_A, G_COUP = 3.0, 1.8
T_ENT, T_FREE, TAU = 0.40, 0.08, 0.25
ALPHA = 1.0
ALPHA_F, G_C, T_LINK = 1.5, 0.1, 0.8
N_TROTTER = 25


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _pauli():
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    i2 = np.eye(2, dtype=np.complex128)
    return sx, sz, sp, sm, i2


def _kron(ops):
    out = ops[0]
    for a in ops[1:]:
        out = np.kron(out, a)
    return out


def _spin(op, site: int):
    _, _, _, _, i2 = _pauli()
    ops = [i2] * NQ
    ops[site] = op
    return _kron(ops)


def build_ops():
    sx, sz, sp, sm, i2 = _pauli()
    hs = np.zeros((D_SPIN, D_SPIN), dtype=np.complex128)
    for i in range(NQ - 1):
        hs -= J * (_spin(sz, i) @ _spin(sz, i + 1))
    for i in range(NQ):
        hs -= HFIELD * _spin(sx, i)

    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    ad = a.conj().T
    nf = ad @ a
    eye_s = np.eye(D_SPIN, dtype=np.complex128)
    eye_f = np.eye(NF, dtype=np.complex128)
    h_mod = np.kron(hs, eye_f) + np.kron(eye_s, OMEGA_A * nf)

    hint = np.zeros((D, D), dtype=np.complex128)
    for s in (0, 2):
        sm_s = np.kron(_spin(sm, s), eye_f)
        sp_s = np.kron(_spin(sp, s), eye_f)
        hint += G_COUP * (sm_s @ np.kron(eye_s, ad) + sp_s @ np.kron(eye_s, a))
    n_op = np.kron(eye_s, nf)

    x3 = np.kron(_spin(sx, 3), eye_f)
    z3 = np.kron(_spin(sz, 3), eye_f)
    x0 = np.kron(_spin(sx, 0), eye_f)
    z0 = np.kron(_spin(sz, 0), eye_f)
    ad_m = np.kron(eye_s, ad)
    a_m = np.kron(eye_s, a)
    h_link = ALPHA_F * (np.kron(x3, x0) + 0.4 * np.kron(z3, z0))
    h_link = h_link + G_C * (np.kron(ad_m, a_m) + np.kron(a_m, ad_m))
    return h_mod, hint, n_op, h_link


def prepare_psi(h_mod, hint):
    sx, sz, sp, sm, i2 = _pauli()
    eye_f = np.eye(NF, dtype=np.complex128)
    psi = np.zeros(D, dtype=np.complex128)
    psi[0] = 1.0
    x0 = np.kron(_spin(sx, 0), eye_f)
    x2 = np.kron(_spin(sx, 2), eye_f)
    psi = x2 @ (x0 @ psi)
    xx = np.kron(_spin(sx, 1) @ _spin(sx, 3), eye_f)
    zz = np.kron(_spin(sz, 1) @ _spin(sz, 3), eye_f)
    psi = expm(-1j * T_ENT * ALPHA * (xx + 0.4 * zz)) @ psi
    psi = expm(-1j * T_FREE * h_mod) @ psi
    psi = expm(-1j * TAU * (h_mod + hint)) @ psi
    psi = psi / norm(psi)
    return psi


# -------------------- MPS --------------------
def product_mps(psi: np.ndarray, n: int, dtype) -> List[np.ndarray]:
    t = psi.astype(dtype).reshape(1, D, 1)
    return [t.copy() for _ in range(n)]


def apply_site(mps: List[np.ndarray], u: np.ndarray, i: int) -> None:
    # A[l, p, r]; U[p_out, p_in]
    mps[i] = np.einsum("op,lpr->lor", u, mps[i], optimize=True)


def apply_bond(
    mps: List[np.ndarray],
    gate4: np.ndarray,
    i: int,
    chi_max: int,
    cutoff: float = 1e-12,
) -> float:
    """gate4: (p_out, q_out, p_in, q_in). Returns discarded weight ~ sum s^2 dropped."""
    a, b = mps[i], mps[i + 1]
    theta = np.einsum("lpr,rqs->lpqs", a, b, optimize=True)
    theta = np.einsum("pqPQ,lPQr->lpqr", gate4, theta, optimize=True)
    cl, d1, d2, cr = theta.shape
    mat = theta.reshape(cl * d1, d2 * cr)
    us, ss, vh = svd(mat, full_matrices=False)
    if ss.size == 0:
        return 0.0
    keep = np.where(ss > cutoff * ss[0])[0]
    if keep.size > chi_max:
        keep = keep[:chi_max]
    disc = float(np.sum(ss[keep.size :] ** 2)) if keep.size < ss.size else 0.0
    us, ss, vh = us[:, keep], ss[keep], vh[keep, :]
    chi = keep.size
    mps[i] = us.reshape(cl, d1, chi)
    mps[i + 1] = (np.diag(ss) @ vh).reshape(chi, d2, cr)
    return disc


def mps_normalize(mps: List[np.ndarray]) -> None:
    left = np.ones((1, 1), dtype=mps[0].dtype)
    for a in mps:
        left = np.einsum("ab,asc,bsd->cd", left, a, np.conj(a), optimize=True)
    nrm = float(np.sqrt(max(np.real(left[0, 0]), 0.0))) + 1e-30
    mps[0] = mps[0] / nrm


def expect_site_seq(mps: List[np.ndarray], op: np.ndarray, i: int) -> float:
    """Sequential contraction — avoids one huge einsum."""
    left = np.ones((1, 1), dtype=mps[0].dtype)
    for j in range(i):
        a = mps[j]
        left = np.einsum("ab,asc,bsd->cd", left, a, np.conj(a), optimize=True)
    right = np.ones((1, 1), dtype=mps[0].dtype)
    for j in range(len(mps) - 1, i, -1):
        a = mps[j]
        right = np.einsum("cd,asc,bsd->ab", right, a, np.conj(a), optimize=True)
    a = mps[i]
    t1 = np.einsum("ab,asc->bsc", left, a, optimize=True)
    t2 = np.einsum("bsc,st->btc", t1, op, optimize=True)
    t3 = np.einsum("btc,btd->cd", t2, np.conj(a), optimize=True)
    val = np.einsum("cd,cd->", t3, right, optimize=True)
    return float(np.real(val))


def e_res(mps: List[np.ndarray], n_op: np.ndarray) -> float:
    op = n_op.astype(mps[0].dtype)
    return sum(OMEGA_A * expect_site_seq(mps, op, i) for i in range(len(mps)))



def s_mid_bond(mps: List[np.ndarray]) -> float:
    """
    Bipartite von Neumann entropy across mid cut.
    Mixed-canonical form at the cut; Schmidt weights = singular values.
    O(N * chi^3 * D) memory-safe — never builds D^(N/2).
    """
    n = len(mps)
    cut = n // 2
    if cut < 1 or cut >= n:
        return 0.0
    tensors = [a.copy() for a in mps]

    # Left-canonicalize sites 0 .. cut-1
    for i in range(cut):
        cl, d, cr = tensors[i].shape
        u, s, vh = svd(tensors[i].reshape(cl * d, cr), full_matrices=False)
        tensors[i] = u.reshape(cl, d, u.shape[1])
        sv = np.diag(s) @ vh
        if i + 1 < n:
            tensors[i + 1] = np.einsum("ij,jkl->ikl", sv, tensors[i + 1], optimize=True)

    # Right-canonicalize sites n-1 .. cut+1
    for i in range(n - 1, cut, -1):
        cl, d, cr = tensors[i].shape
        u, s, vh = svd(tensors[i].reshape(cl, d * cr), full_matrices=False)
        tensors[i] = vh.reshape(vh.shape[0], d, cr)
        us = u @ np.diag(s)
        tensors[i - 1] = np.einsum("ijk,kl->ijl", tensors[i - 1], us, optimize=True)

    # Center tensor at cut: Schmidt spectrum from SVD of (chi_L, D*chi_R)
    cl, d, cr = tensors[cut].shape
    s = svd(tensors[cut].reshape(cl, d * cr), compute_uv=False)
    p = np.real(s * s)
    p = p[p > 1e-16]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def tebd(
    mps: List[np.ndarray],
    u_half: np.ndarray,
    gate4: np.ndarray,
    n_steps: int,
    chi_max: int,
    verbose: bool = False,
) -> float:
    max_disc = 0.0
    for step in range(n_steps):
        for i in range(len(mps)):
            apply_site(mps, u_half, i)
        for i in range(len(mps) - 1):
            max_disc = max(max_disc, apply_bond(mps, gate4, i, chi_max))
        for i in range(len(mps)):
            apply_site(mps, u_half, i)
        mps_normalize(mps)
        if verbose and (step + 1) % 5 == 0:
            print(f"    step {step+1}/{n_steps}  RSS={rss_gb():.2f} GB", flush=True)
            gc.collect()
    return max_disc


def run_one(n, chi, psi, n_op, h_mod, h_link, dtype, n_steps=25, verbose=False):
    """Rebuild gates for given n_steps so Trotter dt = T_LINK/n_steps."""
    t0 = time.time()
    dt = T_LINK / n_steps
    u_half = expm(-1j * (dt / 2.0) * h_mod).astype(dtype)
    gate4 = expm(-1j * dt * h_link).astype(dtype).reshape(D, D, D, D)
    mps = product_mps(psi, n, dtype)
    e0 = e_res(mps, n_op)
    disc = tebd(mps, u_half, gate4, n_steps, chi, verbose=verbose)
    e1 = e_res(mps, n_op)
    sm = s_mid_bond(mps)
    e_m = float(OMEGA_A * np.real(np.vdot(psi, n_op @ psi)))
    ret = 100.0 * e1 / (n * e_m)
    elapsed = time.time() - t0
    out = dict(
        N=n, chi=chi, n_steps=n_steps,
        E0=e0, E=e1, ret=ret, S_mid=sm,
        max_disc=disc, seconds=elapsed, rss_gb=rss_gb(), E_M=e_m,
    )
    del mps
    gc.collect()
    return out





def atomic_save(path, data):
    """Write JSON safely (temp + rename) so Kaggle kill does not corrupt file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    """
    Error quantification with VALIDATED TEBD core (same as N=2,3 F>0.999999).

    Each test changes ONE control variable only.

    Test A — Trotter isolation:   N=10, chi=64, steps in {25,50,100,200}
    Test B — Truncation isolation: N=10, steps=100, chi in {16,24,32,48,64}
    Test C — N scaling:            chi=48, steps=50, N in {5,10,20}

    Observable primary: E_res (resonator energy).
    Secondary: R_N = E_res/(N E_M), D_N = 100(1-R_N) in percentage points,
               S_mid, max_disc, wall time, RSS.

    IMPORTANT:
      D_N is NOT the error vs unknown exact large-N solution.
      It is the modular deficit relative to N*E_M (exact at N=2,3 = 0).
      Fits are exploratory empirical scalings, not universal laws.
    """
    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working"
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."

    path = os.path.join(out_dir, "error_quantification.json")
    print("=" * 64)
    print("ERROR QUANTIFICATION A/B/C | validated TEBD core")
    print("Primary observable: E_res | secondary: R_N, D_N, S_mid, max_disc")
    print("Output:", path)
    print("=" * 64)

    h_mod, hint, n_op, h_link = build_ops()
    psi = prepare_psi(h_mod, hint)
    e_m = float(OMEGA_A * np.real(np.vdot(psi, n_op @ psi)))
    print(f"E_M = {e_m:.6f}  (single-module deposited resonator energy)", flush=True)

    meta = dict(
        E_M=e_m,
        T_LINK=T_LINK,
        ALPHA_F=ALPHA_F,
        G_C=G_C,
        note="D_N=100*(1-E/(N*E_M)); not exact large-N error",
    )
    results = dict(meta=meta, test_A=[], test_B=[], test_C=[], fits={})

    def pack(r):
        """Full record for reproducibility."""
        return dict(
            N=r["N"], chi=r["chi"], steps=r["n_steps"],
            E_res=r["E"], E0=r["E0"],
            R_N=r["ret"] / 100.0,
            D_N=100.0 - r["ret"],
            ret_pct=r["ret"],
            S_mid=r["S_mid"],
            max_disc=r["max_disc"],
            seconds=r["seconds"],
            rss_gb=r["rss_gb"],
            E_M=r["E_M"],
        )

    # ---------- TEST A ----------
    print("\n" + "=" * 64)
    print("TEST A: Trotter isolation | N=10  chi=64  (truncate error suppressed)")
    print("=" * 64, flush=True)
    for steps in (25, 50, 100, 200):
        r = run_one(10, 64, psi, n_op, h_mod, h_link, np.complex64,
                    n_steps=steps, verbose=True)
        row = pack(r)
        results["test_A"].append(row)
        print(
            f"steps={steps:4d}: E={row['E_res']:.8f}  ret={row['ret_pct']:.6f}%  "
            f"D_N={row['D_N']:.6f}pp  S={row['S_mid']:.4f}  disc={row['max_disc']:.2e}  "
            f"({row['seconds']:.1f}s)",
            flush=True,
        )
        atomic_save(path, results)

    # Richardson-style effective order from consecutive doublings (on E_res)
    A = results["test_A"]
    if len(A) >= 3:
        print("  Richardson p_eff on E_res (pairs of doublings):", flush=True)
        # map steps -> E
        emap = {row["steps"]: row["E_res"] for row in A}
        for n in (25, 50):
            if n in emap and 2 * n in emap and 4 * n in emap:
                d1 = emap[n] - emap[2 * n]
                d2 = emap[2 * n] - emap[4 * n]
                if abs(d2) > 1e-16 and abs(d1) > 1e-16:
                    p_eff = np.log(abs(d1 / d2)) / np.log(2.0)
                    print(f"    n={n},{2*n},{4*n}: p_eff = {p_eff:.3f}", flush=True)
                    results["fits"].setdefault("richardson_A", []).append(
                        dict(n=n, p_eff=float(p_eff))
                    )
                else:
                    print(f"    n={n}: differences too small for p_eff", flush=True)
        atomic_save(path, results)

    # ---------- TEST B ----------
    print("\n" + "=" * 64)
    print("TEST B: Truncation isolation | N=10  steps=100")
    print("=" * 64, flush=True)
    for chi in (16, 24, 32, 48, 64):
        r = run_one(10, chi, psi, n_op, h_mod, h_link, np.complex64,
                    n_steps=100, verbose=True)
        row = pack(r)
        results["test_B"].append(row)
        print(
            f"chi={chi:3d}: E={row['E_res']:.8f}  ret={row['ret_pct']:.6f}%  "
            f"D_N={row['D_N']:.6f}pp  S={row['S_mid']:.4f}  disc={row['max_disc']:.2e}  "
            f"({row['seconds']:.1f}s)",
            flush=True,
        )
        atomic_save(path, results)

    # Delta vs chi=64 reference (finite, not exact)
    B = results["test_B"]
    refE = next((row["E_res"] for row in B if row["chi"] == 64), None)
    if refE is not None:
        print("  |E(chi)-E(64)|:", flush=True)
        deltas = []
        for row in B:
            dE = abs(row["E_res"] - refE)
            print(f"    chi={row['chi']:3d}: dE={dE:.6e}", flush=True)
            deltas.append(dict(chi=row["chi"], dE=dE))
        results["fits"]["delta_vs_chi64"] = deltas
        # power-law only on points with dE > 0 and chi < 64
        xs, ys = [], []
        for d in deltas:
            if d["chi"] < 64 and d["dE"] > 1e-14:
                xs.append(d["chi"])
                ys.append(d["dE"])
        if len(xs) >= 2:
            slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
            alpha = -slope
            Acoef = float(np.exp(intercept))
            print(f"  empirical |dE| ~ A chi^(-alpha): alpha={alpha:.3f}  A={Acoef:.4e}", flush=True)
            results["fits"]["truncation_power"] = dict(alpha=float(alpha), A=Acoef)
        atomic_save(path, results)

    # ---------- TEST C ----------
    print("\n" + "=" * 64)
    print("TEST C: N scaling | chi=48  steps=50")
    print("=" * 64, flush=True)
    for n in (5, 10, 20):
        r = run_one(n, 48, psi, n_op, h_mod, h_link, np.complex64,
                    n_steps=50, verbose=True)
        row = pack(r)
        results["test_C"].append(row)
        print(
            f"N={n:3d}: E={row['E_res']:.8f}  ret={row['ret_pct']:.6f}%  "
            f"D_N={row['D_N']:.6f}pp  S={row['S_mid']:.4f}  disc={row['max_disc']:.2e}  "
            f"({row['seconds']:.1f}s)",
            flush=True,
        )
        atomic_save(path, results)

    C = results["test_C"]
    Ns = np.array([row["N"] for row in C], dtype=float)
    Ds = np.array([row["D_N"] for row in C], dtype=float)
    m = Ds > 1e-10
    if m.sum() >= 2:
        beta, logk = np.polyfit(np.log(Ns[m]), np.log(Ds[m]), 1)
        print(f"  exploratory D_N ~ k N^beta: beta={beta:.3f}  k={np.exp(logk):.4e}", flush=True)
        results["fits"]["N_scaling"] = dict(beta=float(beta), k=float(np.exp(logk)))
    else:
        print("  D_N too small for N-scaling fit (near modular reference)", flush=True)
        results["fits"]["N_scaling"] = dict(note="D_N too small")
    atomic_save(path, results)

    print("\n" + "=" * 64)
    print("DONE — interpret fits as exploratory; D_N is modular deficit, not exact error.")
    print("Saved:", path)


main()
