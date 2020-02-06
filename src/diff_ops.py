import numpy as np


def ddx_bwd(f, dx):
    result = np.zeros_like(f)
    result[:, 1:] = (f[:, 1:] - f[:, :-1]) / dx
    return result


def ddy_bwd(f, dy):
    result = np.zeros_like(f)
    result[1:, :] = (f[1:, :] - f[:-1, :]) / dy
    return result


def ddx(f, dx):
    result = np.zeros_like(f)
    result[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0 / dx
    return result


def ddy(f, dy):
    result = np.zeros_like(f)
    result[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0 / dy
    return result


def laplacian(f, dx, dy):
    result = np.zeros_like(f)
    result[1:-1, 1:-1] = (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx / dx \
                         + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy / dy
    return result


def div(u, v, dx, dy):
    return ddx(u, dx) + ddy(v, dy)
