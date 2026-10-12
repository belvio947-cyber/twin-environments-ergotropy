# Appendix E — Measurement program scripts (Paper 2)

Public companion to:
*Modular Quantum Sources for a Conceptual Timing Layer...* (Paper 2 final).

Repository: https://github.com/belvio947-cyber/twin-environments-ergotropy

## Scripts

| File | Paper item | What it does |
|------|------------|--------------|
| `timing_hypothesis_correlator.py` | E.1 baseline | H0 independent, H1 shared latent, classical control, Delta_sigma heatmap |
| `timing_ABC_intervention.py` | E.2–E.4 | A/B/C match, alpha_f intervention, bootstrap power, k discussion |
| `timing_signflip_N2.py` | E.5 | Unitary N=2: frontier correlators for +alpha_f and -alpha_f |
| `code_twin_environments_stable.py` | Table 1 (inherited) | Twin production N=2..5 |
| `robustness_suite.py` | Twin validation | RK4, Fock, parameter scan, negativity |
| `code_channel_diagnosis_N2.py` | Twin channels | FS vs BS |

## Requirements

```
numpy
scipy
matplotlib  # only for correlator figures
```

## Run (order suggested)

```bash
python timing_hypothesis_correlator.py
python timing_ABC_intervention.py
python timing_signflip_N2.py
```

## Important (adversarial)

- These scripts define a **measurement program**. They do **not** demonstrate laboratory quantum timing advantage.
- Control B matching: equalize **marginals of t_i**, **covariance structure of observables**, and optional classical shared fraction — **not** the full histogram of Delta-t if D = std(Delta-t) is the test metric (see Paper 2 corrections).
- Hilbert dimension for N=5: each module is 4 spins x 2 Fock levels = 32; chain dimension = 32^5 = 33,554,432.

## Model units

Energies and times are **dimensionless model units**, not calibrated SI joules or seconds.
