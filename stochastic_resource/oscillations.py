"""Noisy oscillating tragedy of the commons (harmony poor / prisoner's dilemma rich).
The resource follows its exact birth-death chain; between events x follows its ODE
(RK4) at the current resource level; s = 1.
(1) Fraction of runs in which the resource is exhausted, K = 100, values of eps in
    Fig. 7, no immigration; and K = 1000 at eps = 2.5.
(2) K = 1000, eps = 2.5 and 4: standard deviation of x and the angular frequency of
    the spectral peak of x(t)."""
import numpy as np
import numba
import model as M

DL0, DH0, DL1, DH1 = 3.0, 1.0, 2.0, 3.0
LAM, MU, Q, EL, EH, TH = M.LAM, M.MU, M.Q, M.EL, M.EH, 0.5
X0, M0 = 0.4284, 0.4193                         # interior fixed point


@numba.njit
def fx(x, m):
    mh = min(m, 1.0)
    a = (1 - mh) * DL0 - mh * DL1
    b = (1 - mh) * DH0 - mh * DH1
    return x * (1 - x) * (a * x + b * (1 - x))


@numba.njit
def rk4(x, m, tau):
    n = int(tau / 0.01) + 1
    h = tau / n
    for _ in range(n):
        k1 = fx(x, m)
        k2 = fx(x + 0.5 * h * k1, m)
        k3 = fx(x + 0.5 * h * k2, m)
        k4 = fx(x + h * k3, m)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


@numba.njit
def run(K, eps, T, dt, seed):
    """Returns x sampled every dt (nan after exhaustion) and the exhaustion time (-1 if none)."""
    np.random.seed(seed)
    ns = int(T / dt) + 1
    xs = np.full(ns, np.nan)
    t, x, Mr, k = 0.0, X0, int(round(M0 * K)), 1
    xs[0] = x
    while k < ns:
        e = EL * x + EH * (1 - x)
        br = eps * Mr * (LAM - (1 - TH) * Q * e)
        dr = eps * Mr * (MU + (LAM - MU) * Mr / K + TH * Q * e)
        tev = t - np.log(np.random.random()) / (br + dr)
        while k < ns and k * dt <= tev:
            x = rk4(x, Mr / K, k * dt - t)
            t = k * dt
            xs[k] = x
            k += 1
        if k >= ns:
            break
        x = rk4(x, Mr / K, tev - t)
        t = tev
        Mr += 1 if np.random.random() * (br + dr) < br else -1
        if Mr == 0:
            return xs, t
    return xs, -1.0


def peak_frequency(segments, dt, L):
    P, w = 0, np.hanning(L)
    for x in segments:
        for i in range(0, len(x) - L + 1, L):
            y = (x[i:i + L] - x[i:i + L].mean()) * w
            P = P + np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(L, dt)
    return 2 * np.pi * f[1 + np.argmax(P[1:])]


if __name__ == "__main__":
    for K, eps, nrun, T in [(100, 0.5, 300, 1000.0), (100, 1.8, 300, 1000.0),
                            (100, 2.0, 300, 1000.0), (1000, 2.5, 100, 300.0)]:
        ext = np.array([run(K, eps, T, 1.0, 1000 * int(10 * eps) + K + i)[1] for i in range(nrun)])
        hit = ext[ext > 0]
        print(f"K={K} eps={eps}: exhausted by t=500 in {np.mean((ext > 0) & (ext <= 500)):.2f} of {nrun} runs; "
              f"by t={T:.0f} in {len(hit)}" + (f", median time {np.median(hit):.0f}" if len(hit) else ""))
    for eps in (2.5, 4.0):
        segs = []
        for i in range(4):
            xs, e = run(1000, eps, 40000.0, 0.1, 7919 + 101 * int(10 * eps) + i)
            segs.append(xs[5000:])
        X = np.concatenate(segs)
        print(f"K=1000 eps={eps}: sd(x)={np.nanstd(X):.3f}, x range [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}], "
              f"spectral peak {peak_frequency(segs, 0.1, 2**15):.3f}")
