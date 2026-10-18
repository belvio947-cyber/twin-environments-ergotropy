#!/usr/bin/env python3
# ============================================================================
# reference_N2_N3_expm.py
#
# Checksum independiente: misma preparacion y H que tebd_kaggle_N20_trotter.py
# Evolucion exacta (expm_multiply) vs TEBD (chi=32, steps=25).
#
# Protocolo (identico al TEBD de produccion):
#   1) psi_M por modulo: X0,X2 -> entangle intra (t_ent, alpha) -> free H_mod
#      -> transfer (H_mod+Hint, tau)
#   2) Estado inicial cadena: product_i |psi_M>
#   3) Durante t_link: H = sum_i H_mod_i + sum_i H_link_{i,i+1}
#      (H_mod SIN Hint; link = alpha_f(X3X0+0.4 Z3Z0) + g_c beam-splitter)
#
# Uso Kaggle:
#   %%writefile reference_N2_N3_expm.py
#   !python reference_N2_N3_expm.py
# ============================================================================
from __future__ import annotations

import gc
import json
import os
import time
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm, svd
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

# ---- mismos parametros que TEBD produccion ----
J, HFIELD = 1.0, 2.0
NQ, NF = 4, 2
D_SPIN = 2**NQ
D = D_SPIN * NF  # 32
OMEGA_A, G_COUP = 3.0, 1.8
T_ENT, T_FREE, TAU = 0.40, 0.08, 0.25
ALPHA = 1.0
ALPHA_F, G_C, T_LINK = 1.5, 0.1, 0.8
N_TROTTER = 25
CHI_TEBD = 32


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
    """Identico a tebd_kaggle_N20_trotter.prepare_psi"""
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


def embed_site(op: np.ndarray, site: int, n_mod: int) -> sparse.csr_matrix:
    """op (D x D) on module `site` in chain of n_mod modules."""
    left = sparse.eye(D**site, dtype=np.complex128, format="csr")
    mid = sparse.csr_matrix(op)
    right = sparse.eye(D ** (n_mod - site - 1), dtype=np.complex128, format="csr")
    return sparse.kron(sparse.kron(left, mid), right, format="csr")


def embed_bond(op2: np.ndarray, site: int, n_mod: int) -> sparse.csr_matrix:
    """op2 (D*D x D*D) on modules (site, site+1)."""
    left = sparse.eye(D**site, dtype=np.complex128, format="csr")
    mid = sparse.csr_matrix(op2)
    right = sparse.eye(D ** (n_mod - site - 2), dtype=np.complex128, format="csr")
    return sparse.kron(sparse.kron(left, mid), right, format="csr")


def build_full_H(n_mod: int, h_mod: np.ndarray, h_link: np.ndarray) -> sparse.csr_matrix:
    H = sparse.csr_matrix((D**n_mod, D**n_mod), dtype=np.complex128)
    for i in range(n_mod):
        H = H + embed_site(h_mod, i, n_mod)
    for i in range(n_mod - 1):
        H = H + embed_bond(h_link, i, n_mod)
    return H


def product_state(psi_m: np.ndarray, n_mod: int) -> np.ndarray:
    out = psi_m.copy()
    for _ in range(n_mod - 1):
        out = np.kron(out, psi_m)
    return out


def e_res_full(psi: np.ndarray, n_op: np.ndarray, n_mod: int) -> float:
    e = 0.0
    for i in range(n_mod):
        op = embed_site(n_op, i, n_mod)
        e += OMEGA_A * float(np.real(np.vdot(psi, op @ psi)))
    return e


def s_mid_full(psi: np.ndarray, n_mod: int) -> float:
    """Von Neumann entropy across mid cut (pure state Schmidt)."""
    cut = n_mod // 2
    dim_l = D**cut
    dim_r = D ** (n_mod - cut)
    mat = psi.reshape(dim_l, dim_r)
    s = svd(mat, compute_uv=False)
    p = np.real(s * s)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


# ---- TEBD minimo (mismo que produccion) para comparar ----
def product_mps(psi: np.ndarray, n: int) -> List[np.ndarray]:
    t = psi.reshape(1, D, 1).astype(np.complex128)
    return [t.copy() for _ in range(n)]


def apply_site(mps, u, i):
    mps[i] = np.einsum("op,lpr->lor", u, mps[i], optimize=True)


def apply_bond(mps, gate4, i, chi_max, cutoff=1e-12):
    a, b = mps[i], mps[i + 1]
    theta = np.einsum("lpr,rqs->lpqs", a, b, optimize=True)
    theta = np.einsum("pqPQ,lPQr->lpqr", gate4, theta, optimize=True)
    cl, d1, d2, cr = theta.shape
    mat = theta.reshape(cl * d1, d2 * cr)
    us, ss, vh = svd(mat, full_matrices=False)
    keep = np.where(ss > cutoff * ss[0])[0]
    if keep.size > chi_max:
        keep = keep[:chi_max]
    us, ss, vh = us[:, keep], ss[keep], vh[keep, :]
    chi = keep.size
    mps[i] = us.reshape(cl, d1, chi)
    mps[i + 1] = (np.diag(ss) @ vh).reshape(chi, d2, cr)


def mps_normalize(mps):
    left = np.ones((1, 1), dtype=np.complex128)
    for a in mps:
        left = np.einsum("ab,asc,bsd->cd", left, a, np.conj(a), optimize=True)
    nrm = float(np.sqrt(max(np.real(left[0, 0]), 0.0))) + 1e-30
    mps[0] = mps[0] / nrm


def expect_site_seq(mps, op, i):
    left = np.ones((1, 1), dtype=np.complex128)
    for j in range(i):
        a = mps[j]
        left = np.einsum("ab,asc,bsd->cd", left, a, np.conj(a), optimize=True)
    right = np.ones((1, 1), dtype=np.complex128)
    for j in range(len(mps) - 1, i, -1):
        a = mps[j]
        right = np.einsum("cd,asc,bsd->ab", right, a, np.conj(a), optimize=True)
    a = mps[i]
    t1 = np.einsum("ab,asc->bsc", left, a, optimize=True)
    t2 = np.einsum("bsc,st->btc", t1, op, optimize=True)
    t3 = np.einsum("btc,btd->cd", t2, np.conj(a), optimize=True)
    return float(np.real(np.einsum("cd,cd->", t3, right, optimize=True)))


def e_res_mps(mps, n_op):
    return sum(OMEGA_A * expect_site_seq(mps, n_op, i) for i in range(len(mps)))


def s_mid_bond(mps):
    n = len(mps)
    cut = n // 2
    tensors = [a.copy() for a in mps]
    for i in range(cut):
        cl, d, cr = tensors[i].shape
        u, s, vh = svd(tensors[i].reshape(cl * d, cr), full_matrices=False)
        tensors[i] = u.reshape(cl, d, u.shape[1])
        sv = np.diag(s) @ vh
        if i + 1 < n:
            tensors[i + 1] = np.einsum("ij,jkl->ikl", sv, tensors[i + 1], optimize=True)
    for i in range(n - 1, cut, -1):
        cl, d, cr = tensors[i].shape
        u, s, vh = svd(tensors[i].reshape(cl, d * cr), full_matrices=False)
        tensors[i] = vh.reshape(vh.shape[0], d, cr)
        us = u @ np.diag(s)
        tensors[i - 1] = np.einsum("ijk,kl->ijl", tensors[i - 1], us, optimize=True)
    cl, d, cr = tensors[cut].shape
    s = svd(tensors[cut].reshape(cl, d * cr), compute_uv=False)
    p = np.real(s * s)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def mps_to_state(mps: List[np.ndarray]) -> np.ndarray:
    """Contract MPS to full state vector (N=2,3 only)."""
    t = mps[0]  # (1, D, chi)
    for i in range(1, len(mps)):
        t = np.tensordot(t, mps[i], axes=([-1], [0]))
    # t shape: (1, D, D, ..., D, chi_last) with chi_last=1 ideally
    return np.asarray(t, dtype=np.complex128).reshape(-1)


def tebd_run(psi_m, n_mod, h_mod, h_link, n_op, chi, n_steps):
    dt = T_LINK / n_steps
    u_half = expm(-1j * (dt / 2.0) * h_mod)
    gate4 = expm(-1j * dt * h_link).reshape(D, D, D, D)
    mps = product_mps(psi_m, n_mod)
    for _ in range(n_steps):
        for i in range(n_mod):
            apply_site(mps, u_half, i)
        for i in range(n_mod - 1):
            apply_bond(mps, gate4, i, chi)
        for i in range(n_mod):
            apply_site(mps, u_half, i)
        mps_normalize(mps)
    e = e_res_mps(mps, n_op)
    s = s_mid_bond(mps)
    psi_tb = mps_to_state(mps)
    nrm = float(norm(psi_tb))
    if nrm > 0:
        psi_tb = psi_tb / nrm
    return e, s, psi_tb, nrm


def main():
    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working"
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__)) or "."

    print("=" * 64)
    print("REFERENCE checksum: expm_multiply vs TEBD (same protocol)")
    print("=" * 64)

    h_mod, hint, n_op, h_link = build_ops()
    psi_m = prepare_psi(h_mod, hint)
    e_m = float(OMEGA_A * np.real(np.vdot(psi_m, n_op @ psi_m)))
    print(f"E_M = {e_m:.6f}")
    print(f"t_link={T_LINK}  alpha_f={ALPHA_F}  g_c={G_C}")
    print(f"TEBD: chi={CHI_TEBD}  steps={N_TROTTER}")

    results = []

    for n_mod in (2, 3):
        print(f"\n{'='*64}\nN = {n_mod}  dim = {D**n_mod}\n{'='*64}", flush=True)

        # --- exact ---
        t0 = time.time()
        H = build_full_H(n_mod, h_mod, h_link)
        psi0 = product_state(psi_m, n_mod)
        nrm0 = float(norm(psi0))
        e0 = e_res_full(psi0, n_op, n_mod)
        psi_ex = expm_multiply(-1j * H * T_LINK, psi0)
        nrm_ex_raw = float(norm(psi_ex))
        psi_ex = psi_ex / nrm_ex_raw
        e_ex = e_res_full(psi_ex, n_op, n_mod)
        s_ex = s_mid_full(psi_ex, n_mod)
        ret_ex = 100.0 * e_ex / (n_mod * e_m)
        t_ex = time.time() - t0
        print(
            f"EXACT: ||psi0||={nrm0:.10f} ||psi_t||_raw={nrm_ex_raw:.10f}",
            flush=True,
        )
        print(
            f"EXACT expm_multiply: E0={e0:.6f} E={e_ex:.6f} "
            f"ret={ret_ex:.4f}% S={s_ex:.4f} ({t_ex:.1f}s)",
            flush=True,
        )
        del H, psi0
        gc.collect()

        # --- TEBD steps 25 and 50 ---
        row_steps = {}
        for n_steps in (25, 50):
            t0 = time.time()
            e_tb, s_tb, psi_tb, nrm_tb = tebd_run(
                psi_m, n_mod, h_mod, h_link, n_op, CHI_TEBD, n_steps
            )
            ret_tb = 100.0 * e_tb / (n_mod * e_m)
            # fidelity with exact (global phase invariant)
            ov = np.vdot(psi_ex, psi_tb)
            fid = float(np.real(ov * np.conj(ov)))
            t_tb = time.time() - t0
            print(
                f"TEBD chi={CHI_TEBD} steps={n_steps}: E={e_tb:.6f} "
                f"ret={ret_tb:.4f}% S={s_tb:.4f} ||psi||={nrm_tb:.10f} "
                f"F={fid:.10f} ({t_tb:.1f}s)",
                flush=True,
            )
            dE = abs(e_ex - e_tb)
            dS = abs(s_ex - s_tb)
            dret = abs(ret_ex - ret_tb)
            print(
                f"  DELTA |E|={dE:.6e} |ret|={dret:.4f}% |S|={dS:.6f}  "
                f"1-F={1.0-fid:.6e}",
                flush=True,
            )
            row_steps[n_steps] = dict(
                E=e_tb, ret=ret_tb, S=s_tb, F=fid, nrm=nrm_tb,
                abs_E=dE, abs_S=dS, seconds=t_tb,
            )

        results.append(
            dict(
                N=n_mod,
                E_M=e_m,
                exact=dict(
                    E=e_ex, ret=ret_ex, S=s_ex, seconds=t_ex,
                    norm_psi0=nrm0, norm_psi_t_raw=nrm_ex_raw,
                ),
                tebd_by_steps=row_steps,
                chi=CHI_TEBD,
            )
        )
        del psi_ex
        gc.collect()

    path = os.path.join(out_dir, "reference_N2_N3_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 64)
    print("CRITERIOS")
    print("  |dE| << 1e-3, |dS| << 1e-2, F > 0.999 => dinamica TEBD validada")
    print("  Norma ~1 en ambos metodos")
    print("  Si F baja: Trotter/chi o desajuste de H")
    print("Saved:", path)
    print("DONE")


if __name__ == "__main__":
    main()
