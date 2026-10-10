# Finite-resource analysis

Python code for the results on resource persistence, fluctuation size, and noisy
oscillations in "Eco-evolutionary games in noisy environments" (Bodin, Wang & Plotkin).
The figures and analyses of behavior under environmental noise (Figs. 2-4, 6 and S1)
are produced by the notebook in the repository root.

All quantities are computed from the discrete birth-death chain for the resource,
not from its diffusion approximation. Times are in units of 1/(s*epsilon).

| file | output |
|---|---|
| `model.py` | model definitions: rates, strategy rest points, attractors, quasi-stationary distributions, mean collapse times, collapse barriers |
| `fig5_collapse.py` | Fig. 5 (`figures/fig5_collapse.png`) and the barriers and collapse times quoted with it |
| `figS2_noise.py` | SI Fig. S2 (`figures/figS2_noise.png`), K_noise, and the fractions of time at full cooperation (SI Sec. 1.8) |
| `table_barriers.py` | SI Table 2 |
| `table_fast_environment.py` | SI Table 3 and the mean effective environment at K = 100 (SI Sec. 1.10) |
| `collapse_checks.py` | numerical checks in SI Secs. 1.6, 1.7 and 1.9 (quasi-stationary distribution, growth of the collapse time with K, diffusion barrier, stochastic simulation, harvest split, immigration) |
| `hopf.py` | Hopf threshold and eigenvalues (SI Sec. 1.11) |
| `oscillations.py` | resource exhaustion in the oscillating case and the spectrum of x (main text and SI Sec. 1.11) |

Usage:

```
pip install -r requirements.txt
python fig5_collapse.py
```

Each script runs on its own and prints the numbers it produces. `oscillations.py` takes
the longest (several minutes).
