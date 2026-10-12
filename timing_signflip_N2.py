#!/usr/bin/env python3
"""
E.5 Sign-sensitive frontier correlators (N=2 unitary).
Reverses alpha_f and reports <X3 X0>, <Z3 Z0> at t_link=0.8.
Also reports classical common-mode edge C_ij (always positive).

Does NOT claim laboratory quantum advantage.
Requires: numpy, scipy
"""
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix

rng = np.random.default_rng(11)
J, h, Nq, Nf = 1.0, 2.0, 4, 2
omega_A, g_coup = 3.0, 1.8
t_ent, t_free, tau, alpha = 0.40, 0.08, 0.25, 1.0
t_link = 0.8
sx = np.array([[0., 1], [1, 0]], dtype=np.complex128)
sz = np.array([[1., 0], [0, -1]], dtype=np.complex128)
sm = np.array([[0., 0], [1, 0]], dtype=np.complex128)
sp = sm.T
I2 = np.eye(2, dtype=np.complex128)


def kron_list(ops):
    o = ops[0]
    for x in ops[1:]:
        o = np.kron(o, x)
    return o


def spin_mat(op, i, N):
    ops = [I2] * N
    ops[i] = op
    return kron_list(ops)


d_spin = 16
D = 32
Hs = np.zeros((d_spin, d_spin), dtype=np.complex128)
for i in range(Nq - 1):
    Hs -= J * (spin_mat(sz, i, Nq) @ spin_mat(sz, i + 1, Nq))
for i in range(Nq):
    Hs -= h * spin_mat(sx, i, Nq)
a = np.diag(np.sqrt(np.arange(1, Nf)), 1).astype(np.complex128)
ad = a.T
nf = ad @ a
If = np.eye(Nf, dtype=np.complex128)
H_mod = np.kron(Hs, If) + np.kron(np.eye(d_spin), omega_A * nf)

psi0 = np.zeros(D, dtype=np.complex128)
psi0[0] = 1.0
psi_c = np.kron(spin_mat(sx, 2, Nq), If) @ (np.kron(spin_mat(sx, 0, Nq), If) @ psi0)
XX = np.kron(spin_mat(sx, 1, Nq) @ spin_mat(sx, 3, Nq), If)
ZZ = np.kron(spin_mat(sz, 1, Nq) @ spin_mat(sz, 3, Nq), If)
psi_ent = expm(-1j * t_ent * alpha * (XX + 0.4 * ZZ)) @ psi_c
Hint = np.zeros((D, D), dtype=np.complex128)
for s in [0, 2]:
    Hint += g_coup * (
        np.kron(spin_mat(sm, s, Nq), If) @ np.kron(np.eye(d_spin), ad)
        + np.kron(spin_mat(sp, s, Nq), If) @ np.kron(np.eye(d_spin), a)
    )
psi_M = expm(-1j * t_free * H_mod) @ psi_ent
psi_M = expm(-1j * tau * (H_mod + Hint)) @ psi_M
psi_M /= np.linalg.norm(psi_M)

HA = csr_matrix(np.kron(H_mod, np.eye(D)))
HB = csr_matrix(np.kron(np.eye(D), H_mod))
H0 = HA + HB


def embA(op, site):
    return np.kron(np.kron(spin_mat(op, site, Nq), If), np.eye(D))


def embB(op, site):
    return np.kron(np.eye(D), np.kron(spin_mat(op, site, Nq), If))


CXX = embA(sx, 3) @ embB(sx, 0)
CZZ = embA(sz, 3) @ embB(sz, 0)
Hlink = csr_matrix(CXX + 0.4 * CZZ)
psi_prod = np.kron(psi_M, psi_M)


def evolve(af, t):
    return expm_multiply(-1j * t * (H0 + af * Hlink), psi_prod)


def corr(psi, op):
    return float(np.real(np.vdot(psi, op @ psi)))


print("Parameters: J=1,h=2,Nq=4,Nf=2,omega_A=3,g=1.8")
print(f"Prep: alpha={alpha}, t_ent={t_ent}, t_free={t_free}, tau={tau}")
print(f"Link evaluation: t_link={t_link}")
print()
for af in [+1.5, -1.5]:
    psi = evolve(af, t_link)
    print(f"alpha_f={af:+.1f}: <X3X0>={corr(psi, CXX):+.6f}  <Z3Z0>={corr(psi, CZZ):+.6f}")

# Classical common-mode edges (always positive C)
def classical_C(n=30000):
    sh = rng.normal(0, 0.134, n)
    t1 = -(sh + rng.normal(0, 0.148, n))
    t2 = -(sh + rng.normal(0, 0.148, n))
    return float(np.corrcoef(t1, t2)[0, 1])


print(f"\nClassical common-mode C_ij(+label)={classical_C():+.4f}")
print(f"Classical common-mode C_ij(-label)={classical_C():+.4f}")
print("Classical correlation stays positive (no sign flip).")
print("DONE")
