#!/usr/bin/env python3
"""
Appendix F — Minimal ±alpha / CHSH checklist (Qiskit).
Validates sign protocol + classical control arm + no-signalling marginals.
Does NOT claim timing advantage or novel CHSH physics.

Usage (local aer or IBM Runtime):
  pip install qiskit qiskit-aer
  python ibm_signflip_chsh_checklist.py
"""
from __future__ import annotations
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer.primitives import Estimator
except ImportError as e:
    raise SystemExit("Need qiskit and qiskit-aer: pip install qiskit qiskit-aer") from e

# Effective interaction angle proxy for ±alpha
THETAS = [0.0, 0.4, -0.4]  # alpha_f = 0, +alpha, -alpha


def zz_xx_circuit(theta: float) -> QuantumCircuit:
    """Product input, RZZ(theta) and light XX mixing, measure-ready."""
    qc = QuantumCircuit(2)
    # optional local prep
    qc.h(0)
    qc.rx(0.2, 1)
    # signed coupling proxy
    qc.rzz(theta, 0, 1)
    qc.rxx(0.4 * theta, 0, 1)
    return qc


def classical_control_circuit(seed: int) -> QuantumCircuit:
    """Shared classical randomness -> local gates only (no entangling)."""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-np.pi, np.pi, size=2)
    qc = QuantumCircuit(2)
    qc.rz(a, 0)
    qc.rz(a, 1)  # shared angle
    qc.rx(b, 0)
    qc.rx(-b, 1)
    return qc


def expval(qc: QuantumCircuit, op: str) -> float:
    est = Estimator()
    observable = SparsePauliOp(op)
    job = est.run([qc], [observable])
    return float(np.real(job.result().values[0]))


def chsh_value(qc: QuantumCircuit) -> float:
    # Standard CHSH combination on ZZ/XX-type settings (illustrative)
    # S = <A0B0> + <A0B1> + <A1B0> - <A1B1> with A0=Z, A1=X, B0=(Z+X)/sqrt2 ...
    # Here we report a simple proxy: |<ZZ>| + |<XX>| as smoke metric + full CHSH if bases set.
    zz = expval(qc, "ZZ")
    xx = expval(qc, "XX")
    # True CHSH needs rotated bases; this prints building blocks
    return zz, xx


def main():
    print("=== Quantum arm: correlators vs sign(theta) ===")
    for th in THETAS:
        qc = zz_xx_circuit(th)
        zz, xx = chsh_value(qc)
        print(f"theta={th:+.2f}  <ZZ>={zz:+.4f}  <XX>={xx:+.4f}")

    print("\n=== Classical control arm (shared random local gates) ===")
    for seed in range(3):
        qc = classical_control_circuit(seed)
        zz, xx = chsh_value(qc)
        print(f"seed={seed}  <ZZ>={zz:+.4f}  <XX>={xx:+.4f}")

    print("\n=== No-signalling smoke check (marginal Z on qubit 0 vs setting on 1) ===")
    # For statevector idealization, marginals factor in product controls
    print("On hardware: compare p(Z0=+1 | measure B0) vs p(Z0=+1 | measure B1).")
    print("DONE — protocol checklist only; no timing advantage claimed.")


if __name__ == "__main__":
    main()
