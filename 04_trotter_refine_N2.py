#!/usr/bin/env python3
"""
trotter_refine_N2.py
Second-order Trotter refinement of mid-cut S at N=2.
Gaps: 5.8e-4 (25), 1.4e-4 (50), 3.6e-5 (100), 9.0e-6 (200) ~ 1/n_steps^2.
Independent run used in Twin Paper abstract (distinct from TEBD production 1.2989).
"""
import numpy as np
from scipy.linalg import expm

J, h = 1.0, 2.0
Nq, Nf = 4, 2
D_spin, D = 16, 32
omega_A, g_coup = 3.0, 1.8
t_ent, t_free, tau = 0.40, 0.08, 0.25
alpha = 1.0
alpha_f, g_c, t_link = 1.5, 0.1, 0.8
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


def S_mid(psi, dA=D):
    M = psi.reshape(dA, -1)
    sv = np.linalg.svd(M, compute_uv=False)
    p = sv ** 2
    p = p[p > 1e-16]
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


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
psi_M = psi / np.linalg.norm(psi)

X3 = np.kron(spin_mat(sx, 3, Nq), If)
X0 = np.kron(spin_mat(sx, 0, Nq), If)
Z3 = np.kron(spin_mat(sz, 3, Nq), If)
Z0 = np.kron(spin_mat(sz, 0, Nq), If)
a_m = np.kron(eye_s, a_f)
ad_m = np.kron(eye_s, ad_f)
H_link = alpha_f * (np.kron(X3, X0) + 0.4 * np.kron(Z3, Z0)) + g_c * (
    np.kron(ad_m, a_m) + np.kron(a_m, ad_m)
)
H_free = np.kron(H_mod, np.eye(D)) + np.kron(np.eye(D), H_mod)
H2 = H_free + H_link

psi2 = np.kron(psi_M, psi_M)
S_exact = S_mid(expm(-1j * t_link * H2) @ psi2)
print(f"Exact S = {S_exact:.10f}")
print(f"{'steps':>6}  {'S':>12}  {'gap':>12}")

for nsteps in (25, 50, 100, 200):
    dt = t_link / nsteps
    Uf = expm(-1j * H_free * dt / 2)
    Ul = expm(-1j * H_link * dt)
    psi = psi2.copy()
    for _ in range(nsteps):
        psi = Uf @ (Ul @ (Uf @ psi))
    Ss = S_mid(psi)
    gap = S_exact - Ss
    print(f"{nsteps:6d}  {Ss:12.10f}  {gap:12.6e}")
