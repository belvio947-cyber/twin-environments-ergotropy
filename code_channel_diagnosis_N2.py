#!/usr/bin/env python3
# ============================================================================
# N=2 channel diagnosis: product / BS-only / frontier-spin / hybrid
# Reproduces Table 3 of the twin-environments paper
# ============================================================================
import numpy as np
from scipy.linalg import expm

J, h = 1.0, 2.0
Nq, Nf = 4, 2
D, d_spin = 32, 16
omega_A, g_coup = 3.0, 1.8
t_ent, t_free, tau = 0.40, 0.08, 0.25
alpha = 1.0
DT = np.complex128

sx = np.array([[0,1],[1,0]], dtype=DT)
sz = np.array([[1,0],[0,-1]], dtype=DT)
sp = np.array([[0,1],[0,0]], dtype=DT)
sm = np.array([[0,0],[1,0]], dtype=DT)
I2 = np.eye(2, dtype=DT)

def kron_list(ops):
    o = ops[0]
    for x in ops[1:]:
        o = np.kron(o, x)
    return o

def spin_mat(op, i, N):
    ops = [I2]*N; ops[i] = op
    return kron_list(ops)

Hs = np.zeros((d_spin, d_spin), dtype=DT)
for i in range(Nq-1):
    Hs -= J*(spin_mat(sz,i,Nq)@spin_mat(sz,i+1,Nq))
for i in range(Nq):
    Hs -= h*spin_mat(sx,i,Nq)

a_f = np.array([[0,1],[0,0]], dtype=DT)
ad_f = a_f.T.conj(); nf = ad_f@a_f
If = np.eye(Nf, dtype=DT)
H_mod = np.kron(Hs, If) + np.kron(np.eye(d_spin, dtype=DT), omega_A*nf)

psi0 = np.zeros(D, dtype=DT); psi0[0] = 1
X0s = np.kron(spin_mat(sx,0,Nq), If)
X2s = np.kron(spin_mat(sx,2,Nq), If)
psi_c = X2s@(X0s@psi0)
XX = np.kron(spin_mat(sx,1,Nq)@spin_mat(sx,3,Nq), If)
ZZ = np.kron(spin_mat(sz,1,Nq)@spin_mat(sz,3,Nq), If)
psi_ent = expm(-1j*t_ent*alpha*(XX+0.4*ZZ)) @ psi_c

Hint = np.zeros((D,D), dtype=DT)
for s in [0,2]:
    sm_s = np.kron(spin_mat(sm,s,Nq), If)
    sp_s = np.kron(spin_mat(sp,s,Nq), If)
    ad = np.kron(np.eye(d_spin, dtype=DT), ad_f)
    a  = np.kron(np.eye(d_spin, dtype=DT), a_f)
    Hint += g_coup*(sm_s@ad + sp_s@a)
n_op = np.kron(np.eye(d_spin, dtype=DT), nf)
H_loc = H_mod + Hint
psi_M = expm(-1j*t_free*H_mod) @ psi_ent
psi_M = expm(-1j*tau*H_loc) @ psi_M
psi_M /= np.linalg.norm(psi_M)
E_M = float(omega_A*np.real(np.vdot(psi_M, n_op@psi_M)))
print(f"E_M={E_M:.6f}")

# two-module operators (dense, dim 1024)
H_mod_A = np.kron(H_mod, np.eye(D))
H_mod_B = np.kron(np.eye(D), H_mod)
nA = np.kron(n_op, np.eye(D))
nB = np.kron(np.eye(D), n_op)

def embed_A(op, site):
    return np.kron(np.kron(spin_mat(op, site, Nq), If), np.eye(D))
def embed_B(op, site):
    return np.kron(np.eye(D), np.kron(spin_mat(op, site, Nq), If))

X3A, Z3A = embed_A(sx,3), embed_A(sz,3)
X0B, Z0B = embed_B(sx,0), embed_B(sz,0)
aA = np.kron(np.kron(np.eye(d_spin), a_f), np.eye(D))
adA = np.kron(np.kron(np.eye(d_spin), ad_f), np.eye(D))
aB = np.kron(np.eye(D), np.kron(np.eye(d_spin), a_f))
adB = np.kron(np.eye(D), np.kron(np.eye(d_spin), ad_f))

def von_neumann(psi):
    M = psi.reshape(D, D)
    G = M @ M.conj().T
    w = np.linalg.eigvalsh(G)
    w = w[w > 1e-14]
    return float(-np.sum(w*np.log2(w)))

psi_prod = np.kron(psi_M, psi_M)
print(f"product  S={von_neumann(psi_prod):.4f}")

# BS only
for gc in [0.5, 1.0]:
    Hc = gc*(adA@aB + adB@aA)
    psi = expm(-1j*0.25*(H_mod_A+H_mod_B+Hc)) @ psi_prod
    print(f"BS gc={gc:.1f}  S={von_neumann(psi):.4f}")

# frontier-spin only
for af in [1.0, 1.5, 2.0]:
    Hf = af*(X3A@X0B + 0.4*Z3A@Z0B)
    psi = expm(-1j*0.40*(H_mod_A+H_mod_B+Hf)) @ psi_prod
    print(f"FS af={af:.1f}  S={von_neumann(psi):.4f}")

# hybrid best
Hf = 1.5*(X3A@X0B + 0.4*Z3A@Z0B)
Hc = 0.2*(adA@aB + adB@aA)
psi = expm(-1j*0.80*(H_mod_A+H_mod_B+Hf+Hc)) @ psi_prod
print(f"HYB af=1.5 gc=0.2 t=0.8  S={von_neumann(psi):.4f}")
print("DONE")
