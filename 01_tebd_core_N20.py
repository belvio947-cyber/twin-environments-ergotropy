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




def main():
    # Config
    MAX_N = 20
    EXT_NS = [12, 15, 18, 20]   # puntos clave (no todos 11..20 para ahorrar tiempo)
    TROTTER_STEPS = [25, 50, 100]
    CHI_LIST = [16, 32]

    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working"
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."

    print("=" * 64)
    print("TEBD: Trotter conv N=2,3 + extension N=12,15,18,20")
    print("Output:", out_dir)
    print("=" * 64)

    h_mod, hint, n_op, h_link = build_ops()
    psi = prepare_psi(h_mod, hint)
    e_m = float(OMEGA_A * np.real(np.vdot(psi, n_op @ psi)))
    print(f"E_M = {e_m:.6f}  RSS={rss_gb():.2f} GB", flush=True)

    results = []

    # --- 1) Trotter convergence N=2,3 ---
    print("\n=== TROTTER CONVERGENCE N=2,3 (complex128) ===", flush=True)
    for n in (2, 3):
        for n_steps in TROTTER_STEPS:
            for chi in CHI_LIST:
                r = run_one(n, chi, psi, n_op, h_mod, h_link, np.complex128, n_steps=n_steps)
                results.append(r)
                print(
                    f"N={n} steps={n_steps} chi={chi}: "
                    f"ret={r['ret']:.4f}% S={r['S_mid']:.4f} E={r['E']:.6f} "
                    f"disc={r['max_disc']:.2e} ({r['seconds']:.1f}s)",
                    flush=True,
                )
        # delta ret steps 25 vs 100 at chi=32
        r25 = [x for x in results if x.get("N")==n and x.get("n_steps")==25 and x.get("chi")==32]
        r100 = [x for x in results if x.get("N")==n and x.get("n_steps")==100 and x.get("chi")==32]
        if r25 and r100:
            print(
                f"  N={n} chi=32 |ret(25)-ret(100)|={abs(r25[0]['ret']-r100[0]['ret']):.4f}%  "
                f"|S(25)-S(100)|={abs(r25[0]['S_mid']-r100[0]['S_mid']):.4f}",
                flush=True,
            )

    path = os.path.join(out_dir, "tebd_n20_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    # --- 2) Extension ---
    print("\n=== EXTENSION N=12,15,18,20 (complex64, steps=25) ===", flush=True)
    for n in EXT_NS:
        if n > MAX_N:
            continue
        print(f"\n--- N={n} ---", flush=True)
        row = {}
        for chi in CHI_LIST:
            try:
                r = run_one(
                    n, chi, psi, n_op, h_mod, h_link, np.complex64,
                    n_steps=25, verbose=True,
                )
                results.append(r)
                row[chi] = r
                print(
                    f"N={n} chi={chi}: ret={r['ret']:.4f}% S={r['S_mid']:.4f} "
                    f"E={r['E']:.6f} disc={r['max_disc']:.2e} "
                    f"({r['seconds']:.1f}s) RSS={r['rss_gb']:.2f} GB",
                    flush=True,
                )
            except MemoryError:
                print(f"N={n} chi={chi}: MemoryError RSS={rss_gb():.2f}", flush=True)
                results.append(dict(N=n, chi=chi, error="MemoryError"))
                gc.collect()
            except Exception as e:
                print(f"N={n} chi={chi}: {type(e).__name__}: {e}", flush=True)
                results.append(dict(N=n, chi=chi, error=str(e)))
                gc.collect()
            with open(path, "w") as f:
                json.dump(results, f, indent=2)
        if 16 in row and 32 in row:
            print(
                f"  chi conv: |Dret|={abs(row[16]['ret']-row[32]['ret']):.4f}% "
                f"|DS|={abs(row[16]['S_mid']-row[32]['S_mid']):.4f}",
                flush=True,
            )

    print("\n" + "=" * 64)
    print("RESUMEN")
    for r in results:
        if "ret" in r:
            print(
                f"N={r['N']} steps={r.get('n_steps',25)} chi={r['chi']} "
                f"ret={r['ret']:.4f}% S={r['S_mid']:.4f} t={r['seconds']:.1f}s"
            )
        else:
            print(f"N={r.get('N')} ERROR {r.get('error')}")
    print("Saved:", path)
    print("DONE — preliminares TEBD; no afirmar eficiencia termodinamica 99.97%.")


main()
