"""Interior fixed point of the coupled ODEs (harmony poor / prisoner's dilemma rich),
its Jacobian, the Hopf threshold in epsilon, and the eigenvalues above it (s = 1)."""
import numpy as np
from scipy.optimize import brentq
import model as M

g = (3, 1, 2, 3)
m_of_x = lambda x: 1 - M.Q * M.ebar(x) / M.R
xs = brentq(lambda x: M.D(x, m_of_x(x), g), 1e-9, 1 - 1e-9)
ms = m_of_x(xs)
h = 1e-7
Fx = xs * (1 - xs) * (M.D(xs + h, ms, g) - M.D(xs - h, ms, g)) / (2 * h)
Fm = xs * (1 - xs) * (M.D(xs, ms + h, g) - M.D(xs, ms - h, g)) / (2 * h)
Gx = ms * M.Q * (M.EH - M.EL)
Gm = M.R * (1 - 2 * ms) - M.Q * M.ebar(xs)
print(f"fixed point x*={xs:.4f} m*={ms:.4f}")
print(f"Jacobian [[{Fx:.4f}, {Fm:.4f}], [{Gx:.4f} eps, {Gm:.4f} eps]]; det = {Fx*Gm - Fm*Gx:.4f} eps")
eps_c = -Fx / Gm
print(f"trace zero at eps = {eps_c:.4f}")
for eps in (2.0, 2.5, 4.0):
    ev = np.linalg.eigvals(np.array([[Fx, Fm], [eps * Gx, eps * Gm]]))
    lam = ev[np.argmax(ev.imag)]
    print(f"eps={eps}: eigenvalues {lam.real:.4f} +/- {lam.imag:.4f}i, period {2*np.pi/lam.imag:.1f}, "
          f"oscillations per e-fold {lam.imag/(2*np.pi*abs(lam.real)):.1f}")
