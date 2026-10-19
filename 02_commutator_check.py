#!/usr/bin/env python3
"""
commutator_check.py
Verify [H_link, sum_i n_i] = 0 and [H_mod, n] = 0 (JC off during link).
Matches Paper 1 / Twin: number conservation => E_res = N * E_M exactly.
"""
import numpy as np
from numpy.linalg import norm

J, h = 1.0, 2.0
Nq, Nf = 4, 2
D_spin, D = 16, 32
omega_A, g_coup = 3.0, 1.8
alpha_f, g_c = 1.5, 0.1
DT = np.complex128

sx = np.array([[0, 1], [1, 0]], dtype=DT)
sz = np.array([[1, 0], [0, -1]], dtype=DT)
I2 = np.eye(2, dtype=DT)


def kron_list(ops):
    o = ops[0]
    for x in ops[1:]:
        o = np.kron(o, x)
    return o


def spin_mat(op, i, N):
    ops = [I2] * N
    ops[i] = op
    return kron_list(ops)


Hs = np.zeros((D_spin, D_spin), dtype=DT)
for i in range(Nq - 1):
    Hs -= J * (spin_mat(sz, i, Nq) @ spin_mat(sz, i + 1, Nq))
for i in range(Nq):
    Hs -= h * spin_mat(sx, i, Nq)

a_f = np.array([[0, 1], [0, 0]], dtype=DT)
ad_f = a_f.T.conj()
nf = ad_f @ a_f
If = np.eye(Nf, dtype=DT)
eye_s = np.eye(D_spin, dtype=DT)
H_mod = np.kron(Hs, If) + np.kron(eye_s, omega_A * nf)
n_op = np.kron(eye_s, nf)

X3 = np.kron(spin_mat(sx, 3, Nq), If)
X0 = np.kron(spin_mat(sx, 0, Nq), If)
Z3 = np.kron(spin_mat(sz, 3, Nq), If)
Z0 = np.kron(spin_mat(sz, 0, Nq), If)
a_m = np.kron(eye_s, a_f)
ad_m = np.kron(eye_s, ad_f)

H_link = alpha_f * (np.kron(X3, X0) + 0.4 * np.kron(Z3, Z0)) + g_c * (
    np.kron(ad_m, a_m) + np.kron(a_m, ad_m)
)
n_tot = np.kron(n_op, np.eye(D)) + np.kron(np.eye(D), n_op)

comm_link = H_link @ n_tot - n_tot @ H_link
comm_mod = H_mod @ n_op - n_op @ H_mod

# JC does NOT conserve n
from scipy.linalg import expm  # noqa: F401

sp = np.array([[0, 1], [0, 0]], dtype=DT)
sm = np.array([[0, 0], [1, 0]], dtype=DT)
Hint = np.zeros((D, D), dtype=DT)
for s in (0, 2):
    sm_s = np.kron(spin_mat(sm, s, Nq), If)
    sp_s = np.kron(spin_mat(sp, s, Nq), If)
    Hint += g_coup * (sm_s @ np.kron(eye_s, ad_f) + sp_s @ np.kron(eye_s, a_f))
comm_jc = Hint @ n_op - n_op @ Hint

print(f"||[H_link, n_tot]||_F = {norm(comm_link, 'fro'):.3e}")
print(f"||[H_mod,  n    ]||_F = {norm(comm_mod, 'fro'):.3e}")
print(f"||[H_JC,   n    ]||_F = {norm(comm_jc, 'fro'):.3e}  (nonzero: JC off during link)")
assert norm(comm_link, "fro") < 1e-12
assert norm(comm_mod, "fro") < 1e-12
print("PASS: number conservation under link dynamics.")
