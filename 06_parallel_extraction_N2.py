#!/usr/bin/env python3
"""
Parallel extraction (Test F lite): N independent module → target channels.
Like adding amperes in parallel: total extractable energy ~ N * E_channel.

Each channel:
  1) Prepare module (same protocol as TEBD production)
  2) Couple resonator to a target qubit via JC
  3) Evolve t_ext, measure energy on target

Default: N=2 exact (two independent 64-dim systems). Optional N loop for totals.
No claim of full ergotropy — only energy transferred to targets.
"""
import numpy as np
from scipy.linalg import expm
import time, json, os

# ---- module parameters (same as validated TEBD) ----
J, h = 1.0, 2.0
Nq, Nf = 4, 2
D, d_spin = 32, 16
OMEGA_A, G_COUP = 3.0, 1.8
T_ENT, T_FREE, TAU = 0.40, 0.08, 0.25
ALPHA = 1.0
DT = np.complex128

# extraction
G_EXT = 1.2          # JC strength module resonator ↔ target
OMEGA_T = 3.0        # target qubit frequency (matched to resonator)
T_EXT_LIST = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_CHANNELS = 2       # parallel channels

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
    return H_mod, Hint, n_op, a_f, ad_f, nf, If


def prepare_module(H_mod, Hint, n_op):
    psi0 = np.zeros(D, dtype=DT)
    psi0[0] = 1.0
    X0 = np.kron(spin_mat(sx, 0, Nq), np.eye(Nf, dtype=DT))
    X2 = np.kron(spin_mat(sx, 2, Nq), np.eye(Nf, dtype=DT))
    psi = X2 @ (X0 @ psi0)
    XX = np.kron(spin_mat(sx, 1, Nq) @ spin_mat(sx, 3, Nq), np.eye(Nf, dtype=DT))
    ZZ = np.kron(spin_mat(sz, 1, Nq) @ spin_mat(sz, 3, Nq), np.eye(Nf, dtype=DT))
    psi = expm(-1j * T_ENT * ALPHA * (XX + 0.4 * ZZ)) @ psi
    psi = expm(-1j * T_FREE * H_mod) @ psi
    psi = expm(-1j * TAU * (H_mod + Hint)) @ psi
    psi /= np.linalg.norm(psi)
    e_m = float(OMEGA_A * np.real(np.vdot(psi, n_op @ psi)))
    return psi, e_m


def build_channel_hamiltonian(H_mod, a_f, ad_f, nf, If):
    """
    Hilbert space: module (D) ⊗ target (2) = 64.
    H = H_mod ⊗ I + ω_t σz_t/2-style + g (a σ+ + a† σ-)
    Use target as two-level with energy OMEGA_T * |e><e|.
    """
    dim = D * 2
    H = np.zeros((dim, dim), dtype=DT)
    # module free + already includes resonator
    H += np.kron(H_mod, I2)
    # target excitation energy
    n_t = np.array([[0, 0], [0, 1]], dtype=DT)
    H += np.kron(np.eye(D, dtype=DT), OMEGA_T * n_t)
    # JC: a ⊗ σ+  +  a† ⊗ σ-
    a = np.kron(np.eye(d_spin, dtype=DT), a_f)
    ad = np.kron(np.eye(d_spin, dtype=DT), ad_f)
    sigmap = sp  # |e><g|
    sigmam = sm  # |g><e|
    H += G_EXT * (np.kron(a, sigmap) + np.kron(ad, sigmam))
    # observables
    n_res = np.kron(np.kron(np.eye(d_spin, dtype=DT), nf), I2)
    n_tgt = np.kron(np.eye(D, dtype=DT), n_t)
    return H, n_res, n_tgt


def extract_one(psi_m, H_ch, n_res, n_tgt, t_ext):
    """psi_m on module, target starts in ground state."""
    psi = np.kron(psi_m, np.array([1, 0], dtype=DT))
    psi = expm(-1j * t_ext * H_ch) @ psi
    e_res = float(OMEGA_A * np.real(np.vdot(psi, n_res @ psi)))
    e_tgt = float(OMEGA_T * np.real(np.vdot(psi, n_tgt @ psi)))
    return e_res, e_tgt


def main():
    if os.path.isdir("/kaggle/working"):
        out_dir = "/kaggle/working"
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."

    path = os.path.join(out_dir, "parallel_extraction_N2.json")
    print("=" * 64)
    print("PARALLEL EXTRACTION | N channels independent (ampères in parallel)")
    print(f"N_CHANNELS={N_CHANNELS}  G_EXT={G_EXT}  OMEGA_T={OMEGA_T}")
    print("Output:", path)
    print("=" * 64)

    H_mod, Hint, n_op, a_f, ad_f, nf, If = build_module()
    psi_m, e_m = prepare_module(H_mod, Hint, n_op)
    print(f"E_M (deposited per module) = {e_m:.6f}", flush=True)

    H_ch, n_res, n_tgt = build_channel_hamiltonian(H_mod, a_f, ad_f, nf, If)

    rows = []
    print(f"\n{'t_ext':>8} {'E_tgt_1':>10} {'E_tot(N)':>10} {'eta_1':>8} {'eta_tot':>8} {'E_res_left':>10}")
    for t_ext in T_EXT_LIST:
        t0 = time.time()
        # N independent identical channels
        e_tgts = []
        e_ress = []
        for _ in range(N_CHANNELS):
            er, et = extract_one(psi_m, H_ch, n_res, n_tgt, t_ext)
            e_ress.append(er)
            e_tgts.append(et)
        e_tgt_1 = e_tgts[0]
        e_tot = sum(e_tgts)
        e_res_left = sum(e_ress)
        eta_1 = e_tgt_1 / e_m if e_m > 0 else 0.0
        eta_tot = e_tot / (N_CHANNELS * e_m) if e_m > 0 else 0.0
        dt = time.time() - t0
        row = dict(
            t_ext=t_ext,
            E_tgt_one=e_tgt_1,
            E_tgt_total=e_tot,
            E_res_left_total=e_res_left,
            eta_one=eta_1,
            eta_total=eta_tot,
            N=N_CHANNELS,
            E_M=e_m,
            seconds=dt,
        )
        rows.append(row)
        print(
            f"{t_ext:8.2f} {e_tgt_1:10.6f} {e_tot:10.6f} {eta_1:8.4f} {eta_tot:8.4f} {e_res_left:10.6f}",
            flush=True,
        )

    # best t_ext by max total target energy
    best = max(rows, key=lambda r: r["E_tgt_total"])
    print("\n--- best t_ext (max total target energy) ---")
    print(
        f"t_ext={best['t_ext']:.2f}  E_tgt_total={best['E_tgt_total']:.6f}  "
        f"eta_total={best['eta_total']:.4f}  "
        f"(N*E_M={N_CHANNELS*e_m:.6f})",
        flush=True,
    )
    print(
        f"Parallel scaling check: E_tgt_total / E_tgt_one = "
        f"{best['E_tgt_total']/best['E_tgt_one']:.4f}  (expect ~ {N_CHANNELS})",
        flush=True,
    )

    out = dict(
        N_CHANNELS=N_CHANNELS,
        E_M=e_m,
        G_EXT=G_EXT,
        OMEGA_T=OMEGA_T,
        rows=rows,
        best=best,
        note=(
            "Independent channels: total target energy scales ~N. "
            "Per-channel power fixed. Not full ergotropy; energy on target only."
        ),
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", path)
    print("DONE — parallel amperes: more channels → more total energy, same per channel.")


main()
