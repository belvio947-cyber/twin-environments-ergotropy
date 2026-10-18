#!/usr/bin/env python3
"""
CROSSTALK TEST: 2 chains of N=5 modules with optional inter-chain coupling.

H = H_chain1 + H_chain2 + alpha_cross * H_frontier(bond between chains)

When alpha_cross=0 the inter-chain bond is IDENTITY (true independence).
When alpha_cross>0 that bond uses the same frontier form as intra-chain links
scaled by alpha_cross (spin XX+0.4ZZ + weak resonator beam-splitter).

Measure:
  DeltaE = E_res(2 chains, alpha) - 2 * E_res(1 chain)
"""
import numpy as np
from scipy.linalg import expm
from numpy.linalg import svd
import time, os, json, gc

J, h = 1.0, 2.0
Nq, Nf = 4, 2
D, d_spin = 32, 16
OMEGA_A, G_COUP = 3.0, 1.8
T_ENT, T_FREE, TAU = 0.40, 0.08, 0.25
ALPHA = 1.0
ALPHA_F, G_C, T_LINK = 1.5, 0.1, 0.8
N_CHAIN, CHI, N_STEPS = 5, 48, 25
DT = np.complex128

sx = np.array([[0, 1], [1, 0]], dtype=DT)
sz = np.array([[1, 0], [0, -1]], dtype=DT)
sp = np.array([[0, 1], [0, 0]], dtype=DT)
sm = np.array([[0, 0], [1, 0]], dtype=DT)
I2 = np.eye(2, dtype=DT)


def kron_list(ops):
    o = ops[0]
    for x in ops[1:]:
        o = np.kron(o, x)
    return o


def spin_mat(op, i, n):
    ops = [I2] * n
    ops[i] = op
    return kron_list(ops)


def build_module():
    Hs = np.zeros((d_spin, d_spin), dtype=DT)
    for i in range(Nq - 1):
        Hs -= J * (spin_mat(sz, i, Nq) @ spin_mat(sz, i + 1, Nq))
    for i in range(Nq):
        Hs -= h * spin_mat(sx, i, Nq)

    a_f = np.array([[0, 1], [0, 0]], dtype=DT)
    ad_f = a_f.T.conj()
    nf = ad_f @ a_f
    If = np.eye(Nf, dtype=DT)
    H_mod = np.kron(Hs, If) + np.kron(np.eye(d_spin, dtype=DT), OMEGA_A * nf)

    Hint = np.zeros((D, D), dtype=DT)
    for s in (0, 2):
        sm_s = np.kron(spin_mat(sm, s, Nq), If)
        sp_s = np.kron(spin_mat(sp, s, Nq), If)
        ad = np.kron(np.eye(d_spin, dtype=DT), ad_f)
        a = np.kron(np.eye(d_spin, dtype=DT), a_f)
        Hint += G_COUP * (sm_s @ ad + sp_s @ a)

    n_op = np.kron(np.eye(d_spin, dtype=DT), nf)

    psi0 = np.zeros(D, dtype=DT)
    psi0[0] = 1.0
    X0 = np.kron(spin_mat(sx, 0, Nq), If)
    X2 = np.kron(spin_mat(sx, 2, Nq), If)
    psi = X2 @ (X0 @ psi0)
    XX = np.kron(spin_mat(sx, 1, Nq) @ spin_mat(sx, 3, Nq), If)
    ZZ = np.kron(spin_mat(sz, 1, Nq) @ spin_mat(sz, 3, Nq), If)
    psi = expm(-1j * T_ENT * ALPHA * (XX + 0.4 * ZZ)) @ psi
    psi = expm(-1j * T_FREE * H_mod) @ psi
    psi = expm(-1j * TAU * (H_mod + Hint)) @ psi
    psi /= np.linalg.norm(psi)
    e_m = float(OMEGA_A * np.real(np.vdot(psi, n_op @ psi)))

    X3 = np.kron(spin_mat(sx, 3, Nq), If)
    Z3 = np.kron(spin_mat(sz, 3, Nq), If)
    X0m = np.kron(spin_mat(sx, 0, Nq), If)
    Z0m = np.kron(spin_mat(sz, 0, Nq), If)
    a_m = np.kron(np.eye(d_spin, dtype=DT), a_f)
    ad_m = np.kron(np.eye(d_spin, dtype=DT), ad_f)
    H_link = (
        ALPHA_F * (np.kron(X3, X0m) + 0.4 * np.kron(Z3, Z0m))
        + G_C * (np.kron(ad_m, a_m) + np.kron(a_m, ad_m))
    )
    return H_mod, n_op, psi, e_m, H_link


def mps_product(states):
    return [s.reshape(1, D, 1).copy() for s in states]


def mps_norm_sq(mps):
    L = np.ones((1, 1), dtype=DT)
    for A in mps:
        L = np.einsum("ab,asc,bsd->cd", L, A, A.conj(), optimize=True)
    return float(np.real(L[0, 0]))


def mps_normalize(mps):
    n = np.sqrt(mps_norm_sq(mps) + 1e-30)
    mps[0] = mps[0] / n
    return mps


def expect_1site(mps, op, i):
    # left env
    L = np.ones((1, 1), dtype=DT)
    for j in range(i):
        A = mps[j]
        L = np.einsum("ab,asc,bsd->cd", L, A, A.conj(), optimize=True)
    # right env
    R = np.ones((1, 1), dtype=DT)
    for j in range(len(mps) - 1, i, -1):
        A = mps[j]
        R = np.einsum("cd,asc,bsd->ab", R, A, A.conj(), optimize=True)
    A = mps[i]
    T = np.einsum("ab,asc->bsc", L, A, optimize=True)
    T = np.einsum("bsc,st->btc", T, op, optimize=True)
    T = np.einsum("btc,btd->cd", T, A.conj(), optimize=True)
    return float(np.real(np.einsum("cd,cd->", T, R, optimize=True)))


def measure_E_res(mps, n_op):
    return sum(OMEGA_A * expect_1site(mps, n_op, i) for i in range(len(mps)))


def apply_1site(mps, U, i):
    # U: (D,D), A: (chi_l, D, chi_r)
    mps[i] = np.einsum("st,asb->atb", U, mps[i], optimize=True)


def apply_2site(mps, U4, i, chi_max, cutoff=1e-12):
    # U4: (D,D,D,D) acts on physical indices of sites i, i+1
    A, B = mps[i], mps[i + 1]
    Theta = np.tensordot(A, B, axes=([2], [0]))  # (cl, d1, d2, cr)
    Theta = np.einsum("cije,abij->cabe", Theta, U4, optimize=True)
    cl, d1, d2, cr = Theta.shape
    M = Theta.reshape(cl * d1, d2 * cr)
    Us, S, Vh = svd(M, full_matrices=False)
    keep = np.where(S > cutoff * (S[0] if S[0] > 0 else 1.0))[0]
    if len(keep) == 0:
        keep = np.array([0])
    if len(keep) > chi_max:
        keep = keep[:chi_max]
    Us, S, Vh = Us[:, keep], S[keep], Vh[keep, :]
    mps[i] = Us.reshape(cl, d1, -1)
    mps[i + 1] = (np.diag(S) @ Vh).reshape(len(S), d2, cr)


def tebd_two_chains(mps, H_mod, H_link, n_op, alpha_cross, i_cross, chi, n_steps, t_link):
    """
    Intra-chain bonds: full frontier link.
    Inter-chain bond at i_cross:
      alpha_cross == 0 -> identity (no coupling)
      else -> expm(-i dt * alpha_cross/ALPHA_F * H_link) so alpha_cross scales
              the spin-link strength relative to nominal ALPHA_F.
    Actually: H_cross = (alpha_cross/ALPHA_F) * H_link so spin part strength ~ alpha_cross.
    """
    dt = t_link / n_steps
    U_loc = expm(-1j * (dt / 2) * H_mod)
    U_link = expm(-1j * dt * H_link).reshape(D, D, D, D)
    if abs(alpha_cross) < 1e-15:
        U_cross = None  # identity: skip 2-site on that bond
    else:
        # scale full H_link so frontier spin coupling ~ alpha_cross
        scale = alpha_cross / ALPHA_F
        U_cross = expm(-1j * dt * scale * H_link).reshape(D, D, D, D)

    for step in range(n_steps):
        for i in range(len(mps)):
            apply_1site(mps, U_loc, i)
        for i in range(len(mps) - 1):
            if i == i_cross:
                if U_cross is not None:
                    apply_2site(mps, U_cross, i, chi)
                # else: identity — do nothing on inter-chain bond
            else:
                apply_2site(mps, U_link, i, chi)
        for i in range(len(mps)):
            apply_1site(mps, U_loc, i)
        mps_normalize(mps)
        if (step + 1) % 5 == 0:
            print(f"    step {step+1}/{n_steps}", flush=True)
    return measure_E_res(mps, n_op)


def main():
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else (
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
    )
    path = os.path.join(out_dir, "crosstalk_test.json")

    print("=" * 64)
    print(f"CROSSTALK | 2 chains x N={N_CHAIN} | chi={CHI} steps={N_STEPS}")
    print("Inter-chain bond: IDENTITY if alpha_cross=0, else scaled frontier")
    print("=" * 64)

    H_mod, n_op, psi_M, e_m, H_link = build_module()
    print(f"E_M = {e_m:.6f}", flush=True)

    # --- single chain reference ---
    t0 = time.time()
    mps1 = mps_product([psi_M] * N_CHAIN)
    # i_cross out of range => all bonds are intra links
    E_single = tebd_two_chains(
        mps1, H_mod, H_link, n_op, alpha_cross=0.0, i_cross=-1,
        chi=CHI, n_steps=N_STEPS, t_link=T_LINK,
    )
    print(f"1 chain N={N_CHAIN}: E_res={E_single:.6f}  "
          f"ret={100*E_single/(N_CHAIN*e_m):.4f}%  ({time.time()-t0:.1f}s)", flush=True)
    E_expected = 2 * E_single
    print(f"2 chains independent expect: {E_expected:.6f}", flush=True)

    alphas = [0.0, 0.1, 0.5, 1.5]
    results = []
    i_cross = N_CHAIN - 1  # bond between last of chain1 and first of chain2

    for ac in alphas:
        t0 = time.time()
        mps = mps_product([psi_M] * (2 * N_CHAIN))
        E_tot = tebd_two_chains(
            mps, H_mod, H_link, n_op, alpha_cross=ac, i_cross=i_cross,
            chi=CHI, n_steps=N_STEPS, t_link=T_LINK,
        )
        dE = E_tot - E_expected
        dE_pp = 100.0 * dE / E_expected if abs(E_expected) > 1e-15 else 0.0
        dt = time.time() - t0
        print(
            f"alpha_cross={ac:.2f}: E_total={E_tot:.6f}  "
            f"dE={dE:+.6e}  dE%={dE_pp:+.4f}%  ({dt:.1f}s)",
            flush=True,
        )
        results.append(dict(
            alpha_cross=ac, E_total=E_tot, E_expected=E_expected,
            dE=dE, dE_pct=dE_pp, time_s=dt,
        ))
        del mps
        gc.collect()

    out = dict(
        N_chain=N_CHAIN, chi=CHI, steps=N_STEPS, E_M=e_m,
        E_single=E_single, E_expected=E_expected, results=results,
        note=(
            "alpha_cross=0 uses identity on inter-chain bond. "
            "dE~0 at alpha=0 validates independence construction. "
            "Growth of |dE| with alpha bounds crosstalk tolerance."
        ),
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", path)
    print("DONE")


main()
