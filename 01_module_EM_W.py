#!/usr/bin/env python3
"""
module_EM_W.py
Isolated hybrid TFIM+resonator module: deposited energy E_M and local ergotropy W.
Matches Paper 1 (N_Fock=2 production values).
"""
import numpy as np
from scipy.linalg import expm, eigvalsh

J, h = 1.0, 2.0
Nq, Nf = 4, 2
D_spin, D = 16, 32
omega_A, g_coup = 3.0, 1.8
t_ent, t_free, tau = 0.40, 0.08, 0.25
alpha = 1.0
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

psi0 = np.zeros(D, dtype=DT)
psi0[0] = 1.0
X0s = np.kron(spin_mat(sx, 0, Nq), If)
X2s = np.kron(spin_mat(sx, 2, Nq), If)
psi = X2s @ (X0s @ psi0)

XX = np.kron(spin_mat(sx, 1, Nq) @ spin_mat(sx, 3, Nq), If)
ZZ = np.kron(spin_mat(sz, 1, Nq) @ spin_mat(sz, 3, Nq), If)
psi = expm(-1j * t_ent * alpha * (XX + 0.4 * ZZ)) @ psi

Hint = np.zeros((D, D), dtype=DT)
for s in (0, 2):
    sm_s = np.kron(spin_mat(sm, s, Nq), If)
    sp_s = np.kron(spin_mat(sp, s, Nq), If)
    Hint += g_coup * (sm_s @ np.kron(eye_s, ad_f) + sp_s @ np.kron(eye_s, a_f))

psi = expm(-1j * t_free * H_mod) @ psi
psi = expm(-1j * tau * (H_mod + Hint)) @ psi
psi /= np.linalg.norm(psi)

n_op = np.kron(eye_s, nf)
E_M = float(omega_A * np.real(np.vdot(psi, n_op @ psi)))

# Reduced resonator density matrix (Nf=2)
psi_rs = psi.reshape(D_spin, Nf)
rho = psi_rs.T.conj() @ psi_rs
rho = 0.5 * (rho + rho.conj().T)
evals = np.sort(eigvalsh(rho))[::-1]
E_pass = omega_A * float(evals[1])  # passive energy for Nf=2
W = E_M - E_pass

print(f"E_M = {E_M:.12f}")
print(f"W   = {W:.12f}")
print(f"n   = {E_M / omega_A:.12f}")
print(f"rho eigenvalues = {evals}")
