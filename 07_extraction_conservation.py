#!/usr/bin/env python3
"""
Parallel extraction + energy conservation check.
Decomposes energy into: spins | resonator | target | interaction (approx residual).
Verifies total H expectation is constant under unitary evolution.
"""
import numpy as np
from scipy.linalg import expm
import time, json, os

J, h = 1.0, 2.0
Nq, Nf = 4, 2
D, d_spin = 32, 16
OMEGA_A, G_COUP = 3.0, 1.8
T_ENT, T_FREE, TAU = 0.40, 0.08, 0.25
ALPHA = 1.0
G_EXT, OMEGA_T = 1.2, 3.0
T_EXT_LIST = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_CHANNELS = 2
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


def build():
    Hs = np.zeros((d_spin, d_spin), dtype=DT)
    for i in range(Nq - 1):
        Hs -= J * (spin_mat(sz, i, Nq) @ spin_mat(sz, i + 1, Nq))
    for i in range(Nq):
        Hs -= h * spin_mat(sx, i, Nq)

    a_f = np.array([[0, 1], [0, 0]], dtype=DT)
    ad_f = a_f.T.conj()
    nf = ad_f @ a_f
    If = np.eye(Nf, dtype=DT)

    H_spin = np.kron(Hs, If)                     # spins only (in module space)
    H_res = np.kron(np.eye(d_spin, dtype=DT), OMEGA_A * nf)
    H_mod = H_spin + H_res

    Hint = np.zeros((D, D), dtype=DT)
    for s in (0, 2):
        sm_s = np.kron(spin_mat(sm, s, Nq), If)
        sp_s = np.kron(spin_mat(sp, s, Nq), If)
        ad = np.kron(np.eye(d_spin, dtype=DT), ad_f)
        a = np.kron(np.eye(d_spin, dtype=DT), a_f)
        Hint += G_COUP * (sm_s @ ad + sp_s @ a)

    n_op = np.kron(np.eye(d_spin, dtype=DT), nf)
    return H_mod, Hint, H_spin, H_res, n_op, a_f, ad_f, nf


def prepare(H_mod, Hint, n_op):
    psi0 = np.zeros(D, dtype=DT)
    psi0[0] = 1.0
    If = np.eye(Nf, dtype=DT)
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
    return psi, e_m


def channel_ops(H_mod, Hint, H_spin, H_res, a_f, ad_f, nf):
    dim = D * 2
    n_t = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DT)
    a = np.kron(np.eye(d_spin, dtype=DT), a_f)
    ad = np.kron(np.eye(d_spin, dtype=DT), ad_f)

    H_ch = np.kron(H_mod, I2)
    H_ch = H_ch + np.kron(np.eye(D, dtype=DT), OMEGA_T * n_t)
    H_jc_ext = G_EXT * (np.kron(a, sp) + np.kron(ad, sm))
    H_ch = H_ch + H_jc_ext

    # sector operators on full channel space
    Op_spin = np.kron(H_spin, I2)
    Op_res = np.kron(H_res, I2)
    Op_hint = np.kron(Hint, I2)
    Op_tgt = np.kron(np.eye(D, dtype=DT), OMEGA_T * n_t)
    Op_jc = H_jc_ext
    return H_ch, Op_spin, Op_res, Op_hint, Op_tgt, Op_jc


def expect(psi, op):
    return float(np.real(np.vdot(psi, op @ psi)))


def main():
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else (
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
    )
    path = os.path.join(out_dir, "parallel_extraction_conservation.json")

    print("=" * 72)
    print("PARALLEL EXTRACTION + CONSERVATION CHECK")
    print("=" * 72)

    H_mod, Hint, H_spin, H_res, n_op, a_f, ad_f, nf = build()
    psi_m, e_m = prepare(H_mod, Hint, n_op)
    H_ch, Op_spin, Op_res, Op_hint, Op_tgt, Op_jc = channel_ops(
        H_mod, Hint, H_spin, H_res, a_f, ad_f, nf
    )

    print(f"E_M (resonator only after prep) = {e_m:.6f}")
    e_spin0 = expect(psi_m, H_spin)
    e_res0 = expect(psi_m, H_res)
    e_hint0 = expect(psi_m, Hint)
    print(f"After prep (module only): E_spin={e_spin0:.6f}  E_res={e_res0:.6f}  E_Hint={e_hint0:.6f}  sum={e_spin0+e_res0+e_hint0:.6f}")

    psi0 = np.kron(psi_m, np.array([1, 0], dtype=DT))
    e_tot0 = expect(psi0, H_ch)
    print(f"Channel total <H> at t=0: {e_tot0:.6f}")

    print(f"\n{'t':>6} {'E_spin':>10} {'E_res':>10} {'E_Hint':>10} {'E_tgt':>10} {'E_JC':>10} {'sum_sec':>10} {'<H>':>10} {'dH':>10}")
    rows = []
    for t in T_EXT_LIST:
        psi = expm(-1j * t * H_ch) @ psi0
        es = expect(psi, Op_spin)
        er = expect(psi, Op_res)
        eh = expect(psi, Op_hint)
        et = expect(psi, Op_tgt)
        ej = expect(psi, Op_jc)
        sector = es + er + eh + et + ej
        ehtot = expect(psi, H_ch)
        dH = ehtot - e_tot0
        rows.append(dict(
            t=t, E_spin=es, E_res=er, E_Hint=eh, E_tgt=et, E_JC=ej,
            sum_sectors=sector, H_expect=ehtot, dH=dH,
        ))
        print(f"{t:6.2f} {es:10.6f} {er:10.6f} {eh:10.6f} {et:10.6f} {ej:10.6f} {sector:10.6f} {ehtot:10.6f} {dH:10.3e}")

    # parallel scaling at best t by max E_tgt
    best = max(rows, key=lambda r: r["E_tgt"])
    e_tgt_tot = N_CHANNELS * best["E_tgt"]
    print("\n--- parallel at best t (max E_tgt one channel) ---")
    print(f"t={best['t']:.2f}  E_tgt_one={best['E_tgt']:.6f}  E_tgt_total(N={N_CHANNELS})={e_tgt_tot:.6f}")
    print(f"scale factor = {e_tgt_tot/best['E_tgt']:.4f} (expect {N_CHANNELS})")
    print(f"max |dH| over times = {max(abs(r['dH']) for r in rows):.3e}")
    print(f"\nNOTE: E_tgt can exceed E_M because spins+Hint also supply energy.")
    print(f"Conservation: <H> should stay ~ constant (dH ~ 0 within numerical error).")

    with open(path, "w") as f:
        json.dump(dict(E_M=e_m, e_tot0=e_tot0, N=N_CHANNELS, rows=rows, best=best), f, indent=2)
    print("Saved:", path)
    print("DONE")


main()
