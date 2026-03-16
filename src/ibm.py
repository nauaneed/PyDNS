import numpy as np
from scipy import interpolate


def _build_cubic_interpolator(x, y, values):
    return interpolate.RectBivariateSpline(
        y,
        x,
        values,
        kx=min(3, len(y) - 1),
        ky=min(3, len(x) - 1),
    )


def circle(uRHS, vRHS, u, v, dt, x, y, xx, yy, r, R, theta, nx, ny, F, epsilon):
    u_desired = np.zeros_like(u)
    v_desired = np.zeros_like(v)
    interpolate_u = _build_cubic_interpolator(x, y, u + uRHS * dt + F * dt)
    interpolate_v = _build_cubic_interpolator(x, y, v + vRHS * dt)

    for i in range(nx):
        for j in range(ny):
            if epsilon[j, i] != 0:
                x_eval = xx[j, i] + (R - r[j, i]) * np.cos(theta[j, i])
                y_eval = yy[j, i] + (R - r[j, i]) * np.sin(theta[j, i])
                scale = -np.sin(2 * np.pi * (r[j, i] ** 2) / (2 * (R ** 2)))
                u_desired[j, i] = scale * interpolate_u.ev(y_eval, x_eval)
                v_desired[j, i] = scale * interpolate_v.ev(y_eval, x_eval)

    ibm_forcing_u = epsilon * (-uRHS - F + (u_desired - u) / dt)
    ibm_forcing_v = epsilon * (-vRHS + (v_desired - v) / dt)

    return ibm_forcing_u, ibm_forcing_v