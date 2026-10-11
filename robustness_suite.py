#!/usr/bin/env python3
"""Robustness suite for twin-environments paper gaps 1-5 (feasible numerics)."""
import numpy as np
from scipy.linalg import expm
import time

J, h = 1.0, 2.0
Nq = 4
omega_A, g_coup = 3.0, 1.8
t_ent, t_free, tau = 0.40, 0.08, 0.25
alpha = 1.0

sx = np.array([[0,1],[1,0]], dtype=np.complex128)
sz = np.array([[1,0],[0,-1]], dtype=np.complex128)
sp = np.array([[0,1],[0,0]], dtype=np.complex128)
sm = np.array([[0,0],[1,0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

def kron_list(ops):
    o = ops[0]
    for x in ops[1:]:
        o = np.kron(o, x)
    return o

def spin_mat(op, i, N):
    ops = [I2]*N; ops[i] = op
    return kron_list(ops)

def build_module(Nf):
    d_spin = 2**Nq
    D = d_spin * Nf
    Hs = np.zeros((d_spin, d_spin), dtype=np.complex128)
    for i in range(Nq-1):
        Hs -= J*(spin_mat(sz,i,Nq)@spin_mat(sz,i+1,Nq))
    for i in range(Nq):
        Hs -= h*spin_mat(sx,i,Nq)
    # Fock
    a = np.diag(np.sqrt(np.arange(1, Nf)), k=1).astype(np.complex128)
    ad = a.T.conj()
    nf = ad@a
    If = np.eye(Nf, dtype=np.complex128)
    H_mod = np.kron(Hs, If) + np.kron(np.eye(d_spin), omega_A*nf)
    psi0 = np.zeros(D, dtype=np.complex128); psi0[0] = 1
    X0s = np.kron(spin_mat(sx,0,Nq), If)
    X2s = np.kron(spin_mat(sx,2,Nq), If)
    psi_c = X2s@(X0s@psi0)
    XX = np.kron(spin_mat(sx,1,Nq)@spin_mat(sx,3,Nq), If)
    ZZ = np.kron(spin_mat(sz,1,Nq)@spin_mat(sz,3,Nq), If)
    psi_ent = expm(-1j*t_ent*alpha*(XX+0.4*ZZ)) @ psi_c
    Hint = np.zeros((D,D), dtype=np.complex128)
    for s in [0,2]:
        sm_s = np.kron(spin_mat(sm,s,Nq), If)
        sp_s = np.kron(spin_mat(sp,s,Nq), If)
        ad_e = np.kron(np.eye(d_spin), ad)
        a_e  = np.kron(np.eye(d_spin), a)
        Hint += g_coup*(sm_s@ad_e + sp_s@a_e)
    n_op = np.kron(np.eye(d_spin), nf)
    H_loc = H_mod + Hint
    psi_M = expm(-1j*t_free*H_mod) @ psi_ent
    psi_M = expm(-1j*tau*H_loc) @ psi_M
    psi_M /= np.linalg.norm(psi_M)
    EM = float(omega_A*np.real(np.vdot(psi_M, n_op@psi_M)))
    return D, d_spin, H_mod, n_op, psi_M, EM, a, ad, nf, If

# ========== 1. N_Fock truncation ==========
print("=== 1. Resonator truncation N_Fock ===")
for Nf in [2, 3, 4]:
    D, ds, Hm, nop, psiM, EM, *_ = build_module(Nf)
    print(f"  N_Fock={Nf}  D={D}  E_M={EM:.6f}")

# baseline Nf=2
D, d_spin, H_mod, n_op, psi_M, E_M, a_f, ad_f, nf, If = build_module(2)
print(f"Baseline E_M={E_M:.6f}")

# operators for N=2 dense
H_mod_A = np.kron(H_mod, np.eye(D))
H_mod_B = np.kron(np.eye(D), H_mod)
nA = np.kron(n_op, np.eye(D)); nB = np.kron(np.eye(D), n_op)

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

def S_vN(psi):
    M = psi.reshape(D, D)
    G = M @ M.conj().T
    w = np.real(np.linalg.eigvalsh(G))
    w = w[w > 1e-14]
    return float(-np.sum(w*np.log2(w)))

def negativity(psi):
    """Log negativity via partial transpose on B."""
    M = psi.reshape(D, D)  # A, B
    # rho = |psi><psi|
    rho = np.outer(psi, psi.conj())
    # reshape (Da, Db, Da, Db) and transpose B
    rho = rho.reshape(D, D, D, D)
    rho_pt = rho.transpose(0, 3, 2, 1).reshape(D*D, D*D)
    w = np.linalg.eigvalsh(rho_pt)
    # trace norm of PT = sum |w|
    return float(np.sum(np.abs(w)) - 1.0)/2.0  # negativity

def E_tot(psi):
    return float(omega_A*np.real(np.vdot(psi, nA@psi)+np.vdot(psi, nB@psi)))

psi_prod = np.kron(psi_M, psi_M)

# ========== 2. Exact unitary N=2 reference ==========
print("\n=== 2. Exact unitary N=2 (complex128) reference ===")
af, gc, t_link = 1.5, 0.1, 0.8
Hf = af*(X3A@X0B + 0.4*Z3A@Z0B)
Hc = gc*(adA@aB + adB@aA)
Htot = H_mod_A + H_mod_B + Hf + Hc
psi_exact = expm(-1j*t_link*Htot) @ psi_prod
S_ex = S_vN(psi_exact)
N_ex = negativity(psi_exact)
E_ex = E_tot(psi_exact)
print(f"  exact: E={E_ex:.6f}  S={S_ex:.4f}  Neg={N_ex:.4f}")

# ========== 3. RK4 step convergence (N=2, complex128) ==========
print("\n=== 3. RK4 step convergence N=2 (vs exact) ===")

def apply_H(v):
    return Htot @ v

def rk4_dense(psi0, t, nsteps, renorm=True):
    dt = t/nsteps
    y = psi0.copy()
    for _ in range(nsteps):
        k1 = -1j*apply_H(y)
        k2 = -1j*apply_H(y+0.5*dt*k1)
        k3 = -1j*apply_H(y+0.5*dt*k2)
        k4 = -1j*apply_H(y+dt*k3)
        y = y + (dt/6)*(k1+2*k2+2*k3+k4)
        if renorm:
            y /= np.linalg.norm(y)
    if not renorm:
        y /= np.linalg.norm(y)
    return y

for ns in [5, 10, 15, 20, 25, 40, 80]:
    psi_r = rk4_dense(psi_prod, t_link, ns, renorm=True)
    psi_n = rk4_dense(psi_prod, t_link, ns, renorm=False)
    fid_r = abs(np.vdot(psi_exact, psi_r))**2
    fid_n = abs(np.vdot(psi_exact, psi_n))**2
    print(f"  nsteps={ns:3d}  renorm: S={S_vN(psi_r):.4f} E={E_tot(psi_r):.6f} F={fid_r:.6f}  "
          f"no-renorm: F={fid_n:.6f} S={S_vN(psi_n):.4f}")

# ========== 4. complex64 vs complex128 RK4 ==========
print("\n=== 4. complex64 vs complex128 RK4 (nsteps=25) ===")
# complex128 already above
psi64 = rk4_dense(psi_prod.astype(np.complex64).astype(np.complex128), t_link, 25, renorm=True)
# true c64 path: cast H action
def rk4_c64(psi0, t, nsteps):
    dt = np.float32(t/nsteps)
    y = psi0.astype(np.complex64)
    Ht = Htot.astype(np.complex64)
    for _ in range(nsteps):
        k1 = (-1j* (Ht@y)).astype(np.complex64)
        k2 = (-1j* (Ht@(y+0.5*dt*k1))).astype(np.complex64)
        k3 = (-1j* (Ht@(y+0.5*dt*k2))).astype(np.complex64)
        k4 = (-1j* (Ht@(y+dt*k3))).astype(np.complex64)
        y = y + (dt/6)*(k1+2*k2+2*k3+k4)
        y /= np.linalg.norm(y)
    return y.astype(np.complex128)

psi_c64 = rk4_c64(psi_prod, t_link, 25)
print(f"  c128 RK4: S={S_vN(psi_r):.4f} E={E_tot(psi_r):.6f} F_exact={abs(np.vdot(psi_exact,psi_r))**2:.6f}")
print(f"  c64  RK4: S={S_vN(psi_c64):.4f} E={E_tot(psi_c64):.6f} F_exact={abs(np.vdot(psi_exact,psi_c64))**2:.6f}")
print(f"  |S_c64-S_c128|={abs(S_vN(psi_c64)-S_vN(psi_r)):.6f}")

# ========== 5. Parameter scan alpha_f, t_link ==========
print("\n=== 5. Parameter scan (exact N=2) ===")
print(f"{'af':>5} {'t':>5} {'S':>8} {'Neg':>8} {'E':>10} {'ret%':>7}")
E0 = E_tot(psi_prod)
for af in [0.5, 1.0, 1.5, 2.0, 2.5]:
    for t in [0.4, 0.6, 0.8, 1.0]:
        Hf = af*(X3A@X0B + 0.4*Z3A@Z0B)
        Hc = 0.1*(adA@aB + adB@aA)
        psi = expm(-1j*t*(H_mod_A+H_mod_B+Hf+Hc)) @ psi_prod
        S = S_vN(psi); Neg = negativity(psi); E = E_tot(psi)
        print(f"{af:5.1f} {t:5.1f} {S:8.4f} {Neg:8.4f} {E:10.6f} {100*E/E0:7.2f}")

# ========== 6. Why ~1.3 bits: reduced spectrum ==========
print("\n=== 6. Schmidt spectrum at operating point ===")
M = psi_exact.reshape(D, D)
svals = np.linalg.svd(M, compute_uv=False)
p = svals**2
p = p[p > 1e-14]
print(f"  top-8 probabilities: {p[:8]}")
print(f"  effective rank (p>1e-3): {np.sum(p>1e-3)}")
print(f"  S = {-np.sum(p*np.log2(p)):.4f}")
print(f"  max possible log2(D)={np.log2(D):.2f}")

print("\nDONE")
