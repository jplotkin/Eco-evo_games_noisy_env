# Figures and analyses

Python code for all figures and numerical results in "Eco-evolutionary games in noisy
environments" (Bodin, Wang & Plotkin).

Figs. 2-4, 6 and S1 are simulations of the resource level m by its stochastic
differential equation (Euler-Maruyama), with the strategy frequency x either at its
stable rest point (timescale separation, epsilon -> 0) or integrated by its ODE. As in
the original notebook, a resource level that falls to zero is reset to 1/1000 in these
simulations. Fig. 5, SI Fig. S2 and SI Tables 2 and 3 are computed from the discrete
birth-death chain for the resource, in which zero resource is absorbing. Times are in
units of 1/(s*epsilon).

| file | output |
|---|---|
| `model.py` | model definitions: rates, strategy rest points, attractors, quasi-stationary distributions, mean collapse times, collapse barriers |
| `sde.py` | stochastic simulation of the resource coupled to the strategy dynamics; diffusion approximation to the distribution of m |
| `panels.py` | dilemma phase plane and bifurcation diagram panels shared by Figs. 2, 3 and S1 |
| `fig1_schematic.py` | Fig. 1 (`figures/fig1_schematic.pdf`) |
| `fig2_examples.py` | Fig. 2 (`figures/fig2_examples.png`) and the time averages quoted with it |
| `fig3_bifurcations.py` | Fig. 3 (`figures/fig3_bifurcations.png`) |
| `fig4_distributions.py` | Fig. 4 (`figures/fig4_distributions.png`) |
| `fig6_cycles.py` | Fig. 6 (`figures/fig6_cycles.png`) and the mean payoffs shown in its panels |
| `figS1_snowdrift.py` | SI Fig. S1 (`figures/figS1_snowdrift.png`) |
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
the longest (several minutes). `fig2_examples.py`, `fig6_cycles.py` and `figS1_snowdrift.py`
accept `--as-submitted`, which draws the axis labels of the first submitted version.
