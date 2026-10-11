#!/usr/bin/env python3
# ============================================================================
# Entornos gemelos N=2..5 — Versión estable (Kaggle / Colab / local)
# Arquitectura Paper 1: 4 qubits + resonador (N_Fock=2) por módulo
# Frontier-spin chain + weak resonator BS, matrix-free RK4, complex64
# ============================================================================
import numpy as np
from scipy.linalg import expm
import time, gc, resource, os

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

# -------------------- parameters (Paper 1 + twin link) --------------------
J, h = 1.0, 2.0
Nq, Nf = 4, 2
D, d_spin = 32, 16
omega_A, g_coup = 3.0, 1.8
t_ent, t_free, tau = 0.40, 0.08, 0.25
alpha = 1.0
DT = np.complex64

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

# -------------------- single-module operators & state --------------------
Hs = np.zeros((d_spin, d_spin), dtype=DT)
for i in range(Nq - 1):
    Hs -= np.float32(J) * (spin_mat(sz, i, Nq) @ spin_mat(sz, i + 1, Nq))
for i in range(Nq):
    Hs -= np.float32(h) * spin_mat(sx, i, Nq)

a_f = np.array([[0, 1], [0, 0]], dtype=DT)
ad_f = a_f.T.conj()
nf = ad_f @ a_f
If = np.eye(Nf, dtype=DT)
H_mod = (np.kron(Hs, If) + np.kron(np.eye(d_spin, dtype=DT), np.float32(omega_A) * nf)).astype(DT)

psi0 = np.zeros(D, dtype=DT)
psi0[0] = 1.0
X0s = np.kron(spin_mat(sx, 0, Nq), If).astype(DT)
X2s = np.kron(spin_mat(sx, 2, Nq), If).astype(DT)
psi_c = X2s @ (X0s @ psi0)

# CRITICAL: kron(spin@spin, If) — dimensions 16x16 then embed to 32
XX = np.kron(spin_mat(sx, 1, Nq) @ spin_mat(sx, 3, Nq), If)
ZZ = np.kron(spin_mat(sz, 1, Nq) @ spin_mat(sz, 3, Nq), If)
psi_ent = expm((-1j * t_ent * alpha * (XX + 0.4 * ZZ)).astype(np.complex128)) @ psi_c.astype(np.complex128)
psi_ent = psi_ent.astype(DT)

Hint = np.zeros((D, D), dtype=DT)
for s in [0, 2]:
    sm_s = np.kron(spin_mat(sm, s, Nq), If).astype(DT)
    sp_s = np.kron(spin_mat(sp, s, Nq), If).astype(DT)
    ad = np.kron(np.eye(d_spin, dtype=DT), ad_f)
    a = np.kron(np.eye(d_spin, dtype=DT), a_f)
    Hint += np.float32(g_coup) * (sm_s @ ad + sp_s @ a)
n_op = np.kron(np.eye(d_spin, dtype=DT), nf).astype(DT)
H_loc = (H_mod + Hint).astype(DT)

psi_M = expm((-1j * t_free * H_mod).astype(np.complex128)) @ psi_ent.astype(np.complex128)
psi_M = expm((-1j * tau * H_loc).astype(np.complex128)) @ psi_M
psi_M = psi_M.astype(DT)
psi_M /= np.linalg.norm(psi_M)

E_M = float(omega_A * np.real(np.vdot(
    psi_M.astype(np.complex128),
    n_op.astype(np.complex128) @ psi_M.astype(np.complex128))))
print(f"E_M={E_M:.6f}  RSS={rss_mb():.0f}MB", flush=True)

# frontier / resonator ops on one module
X3 = np.kron(spin_mat(sx, 3, Nq), If).astype(DT)
Z3 = np.kron(spin_mat(sz, 3, Nq), If).astype(DT)
X0 = np.kron(spin_mat(sx, 0, Nq), If).astype(DT)
Z0 = np.kron(spin_mat(sz, 0, Nq), If).astype(DT)
a_m = np.kron(np.eye(d_spin, dtype=DT), a_f).astype(DT)
ad_m = np.kron(np.eye(d_spin, dtype=DT), ad_f).astype(DT)

# -------------------- matrix-free apply --------------------
def apply_local(H, psi, axis, N):
    order = [axis] + [i for i in range(N) if i != axis]
    inv = np.argsort(order)
    t = np.transpose(psi, order)
    sh = t.shape
    t = (H @ t.reshape(D, -1)).reshape(sh)
    return np.transpose(t, inv)

def apply_H_N(v, N, af, g_c):
    t = np.asarray(v, dtype=DT).reshape((D,) * N)
    out = np.zeros_like(t)
    for i in range(N):
        out += apply_local(H_mod, t, i, N)
    if af != 0.0:
        for i in range(N - 1):
            tmp = apply_local(X3, t, i, N)
            tmp = apply_local(X0, tmp, i + 1, N)
            out += np.float32(af) * tmp
            tmp = apply_local(Z3, t, i, N)
            tmp = apply_local(Z0, tmp, i + 1, N)
            out += np.float32(0.4 * af) * tmp
    if g_c != 0.0:
        for i in range(N - 1):
            tmp = apply_local(ad_m, t, i, N)
            tmp = apply_local(a_m, tmp, i + 1, N)
            out += np.float32(g_c) * tmp
            tmp = apply_local(a_m, t, i, N)
            tmp = apply_local(ad_m, tmp, i + 1, N)
            out += np.float32(g_c) * tmp
    return out.ravel()

def rk4(psi, N, af, g_c, t, nsteps):
    """RK4 with normalization after EVERY step (required for stability)."""
    dt = np.float32(t / nsteps)
    y = np.asarray(psi, dtype=DT).copy()
    for step in range(nsteps):
        k1 = -1j * apply_H_N(y, N, af, g_c)
        k2 = -1j * apply_H_N(y + 0.5 * dt * k1, N, af, g_c)
        k3 = -1j * apply_H_N(y + 0.5 * dt * k2, N, af, g_c)
        k4 = -1j * apply_H_N(y + dt * k3, N, af, g_c)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y /= np.linalg.norm(y)  # mandatory each step
        if (step + 1) % 5 == 0 or step + 1 == nsteps:
            print(f"    step {step+1}/{nsteps}  RSS={rss_mb():.0f}MB", flush=True)
            gc.collect()
    return y

def energy_total(psi, N):
    t = np.asarray(psi, dtype=DT).reshape((D,) * N)
    E = 0.0
    for i in range(N):
        ni = apply_local(n_op, t, i, N)
        E += omega_A * float(np.real(np.vdot(
            t.ravel().astype(np.complex128),
            ni.ravel().astype(np.complex128))))
    return E

def bipartite_S(psi, N, cut):
    left = D ** cut
    right = D ** (N - cut)
    M = np.asarray(psi, dtype=np.complex128).reshape(left, right)
    if left <= right:
        G = M @ M.conj().T
        w = np.linalg.eigvalsh(G)
    else:
        G = M.conj().T @ M
        w = np.linalg.eigvalsh(G)
    w = np.real(w)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log2(w)))

def run_N(N, af=1.5, g_c=0.1, t_link=0.8, nsteps=20, outdir="."):
    t0 = time.time()
    psi = psi_M.copy()
    for _ in range(1, N):
        psi = np.kron(psi, psi_M)
        gc.collect()
    print(f"\nN={N}  dim={D**N:,}  state={psi.nbytes/1e6:.1f}MB  RSS={rss_mb():.0f}MB", flush=True)
    E0 = energy_total(psi, N)
    S0 = bipartite_S(psi, N, 1)
    print(f"  product E={E0:.5f} (N*EM={N*E_M:.5f})  S(1|r)={S0:.4f}", flush=True)

    psi = rk4(psi, N, af, g_c, t_link, nsteps)

    E1 = energy_total(psi, N)
    S1 = bipartite_S(psi, N, 1)
    Sm = bipartite_S(psi, N, max(1, N // 2))
    elapsed = time.time() - t0
    print(f"  link af={af} gc={g_c}: E={E1:.5f}  S(1|r)={S1:.4f}  "
          f"S(mid)={Sm:.4f}  time={elapsed:.1f}s  RSS={rss_mb():.0f}MB", flush=True)

    line = (f"N={N} E_prod={E0:.6f} E_link={E1:.6f} "
            f"S1={S1:.4f} Smid={Sm:.4f} t={elapsed:.1f}s\n")
    path = os.path.join(outdir, f"twin_N{N}.txt")
    with open(path, "w") as f:
        f.write(line)
    print(f"  saved {path}", flush=True)

    del psi
    gc.collect()
    return E0, E1, S1, Sm

# -------------------- execution --------------------
if __name__ == "__main__":
    # Auto-detect output dir (Kaggle vs local/Colab)
    if os.path.isdir("/kaggle/working"):
        OUT = "/kaggle/working"
    else:
        OUT = "."

    print("=== Entornos gemelos N=2..5 (version estable) ===", flush=True)
    rows = []
    # (N, nsteps) — more steps for larger N improves stability
    config = [
        (2, 25),
        (3, 20),
        (4, 30),
        (5, 25),  # ~20-30 min on Kaggle high-RAM; skip if low memory
    ]
    for N, steps in config:
        try:
            E0, E1, S1, Sm = run_N(N, af=1.5, g_c=0.1, t_link=0.8,
                                   nsteps=steps, outdir=OUT)
            rows.append((N, E0, E1, S1, Sm))
            gc.collect()
        except Exception as e:
            print(f"N={N} FALLO: {type(e).__name__}: {e}", flush=True)
            break

    print("\n========== RESUMEN ==========")
    print(f"{'N':>3} {'E_prod':>10} {'E_link':>10} {'S(1|r)':>8} {'S(mid)':>8}")
    for N, E0, E1, S1, Sm in rows:
        print(f"{N:3d} {E0:10.5f} {E1:10.5f} {S1:8.4f} {Sm:8.4f}")
    print("DONE")
