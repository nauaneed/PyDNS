import numpy as np


def ddx(f, dx, bc):
    result = np.zeros_like(f)
    result[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / 2.0 / dx
    if bc['x'] == 'periodic':
        result[:, -1] = (f[:, 0] - f[:, -2]) / 2.0 / dx
        result[:, 0] = (f[:, 1] - f[:, -1]) / 2.0 / dx
    return result


def ddy(f, dy, bc):
    result = np.zeros_like(f)
    result[1:-1,1 :-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / 2.0 / dy
    return result


def laplacian(f, dx, dy, bc):
    result = np.zeros_like(f)
    result[1:-1, 1:-1] = (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx / dx \
                         + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy / dy
    if bc['x'] == 'periodic':
        result[1:-1, -1] = (f[1:-1, -2] - 2.0 * f[1:-1, -1] + f[1:-1, 0]) / dx / dx \
                           + (f[2:, -1] - 2.0 * f[1:-1, -1] + f[:-2, -1]) / dy / dy

        result[1:-1, 0] = (f[1:-1, -1] - 2.0 * f[1:-1, 0] + f[1:-1, 1]) / dx / dx \
                          + (f[2:, 0] - 2.0 * f[1:-1, 0] + f[:-2, 0]) / dy / dy
    return result


def div(u, v, dx, dy, bc):
    return ddx(u, dx, bc) + ddy(v, dy, bc)


def conv_diff_u(u, v, nu, dx, dy, bc):
    result = - ddx(u * u, dx, bc) - ddy(v * u, dy, bc) + nu * laplacian(u, dx, dy, bc)
    return result


def conv_diff_v(u, v, nu, dx, dy, bc):
    result = - ddx(u * v, dx, bc) - ddy(v * v, dy, bc) + nu * laplacian(v, dx, dy, bc)
    return result
