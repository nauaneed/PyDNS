import numpy as np
from src import derive, projection_method, ibm, ip_op


def rk3(u, v, nx, ny, nu, dx, dy, dt, dpdx, dpdy, epsilon, F, theta, r, R, rho, stepcount,
        saveth_iter, x, y, xx, yy, nx_sp, ny_sp, K):


    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0

    # Step1
    # do the x-momentum RHS
    t = 0
    k1_u = - u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(u, dx, dy)
    k1_v = - u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(v, dx, dy)

    # periodic condition at x=lx
    utemp = np.hstack((u[:, -2:].reshape((ny, 2)), u[:, 0].reshape((ny, 1))))
    vtemp = np.hstack((v[:, -2:].reshape((ny, 2)), v[:, 0].reshape((ny, 1))))
    k1_u[:, -1] = (- u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(
        u, dx, dy))[:, 1]
    k1_v[:, -1] = (- u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(
        v, dx, dy))[:, 1]
    # periodic condition at x=0
    utemp = np.hstack((u[:, -1].reshape((ny, 1)), u[:, :2].reshape((ny, 2))))
    vtemp = np.hstack((v[:, -1].reshape((ny, 1)), v[:, :2].reshape((ny, 2))))
    k1_u[:, 0] = (- u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(u, dx, dy)
                  - dpdx)[:, 1]
    k1_v[:, 0] = (- u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(v, dx, dy)
                  - dpdy)[:, 1]

    t = t + 0.5 * dt

    k2_u = - (u + 0.5 * k1_u * dt) * derive.ddx_bwd((u + 0.5 * k1_u * dt), dx) - (
            v + 0.5 * k1_v * dt) * derive.ddy_bwd((u + 0.5 * k1_u * dt), dy) + nu * derive.laplacian(
        (u + 0.5 * k1_u * dt), dx, dy)
    k2_v = - (u + 0.5 * k1_u * dt) * derive.ddx_bwd((v + 0.5 * k1_v * dt), dx) - (
            v + 0.5 * k1_v * dt) * derive.ddy_bwd((v + 0.5 * k1_v * dt), dy) + nu * derive.laplacian(
        (v + 0.5 * k1_v * dt), dx, dy)

    # periodic condition at x=lx
    utemp = np.hstack(
        ((u + 0.5 * k1_u * dt)[:, -2:].reshape((ny, 2)), (u + 0.5 * k1_u * dt)[:, 0].reshape((ny, 1))))
    vtemp = np.hstack(
        ((v + 0.5 * k1_v * dt)[:, -2:].reshape((ny, 2)), (v + 0.5 * k1_v * dt)[:, 0].reshape((ny, 1))))
    k2_u[:, -1] = (- (u + 0.5 * k1_u * dt) * derive.ddx_bwd((u + 0.5 * k1_u * dt), dx) - (
            v + 0.5 * k1_v * dt) * derive.ddy_bwd((u + 0.5 * k1_u * dt), dy) + nu * derive.laplacian(
        (u + 0.5 * k1_u * dt), dx, dy))[:, 1]
    k2_v[:, -1] = (- (u + 0.5 * k1_u * dt) * derive.ddx_bwd((v + 0.5 * k1_v * dt), dx) - (
            v + 0.5 * k1_v * dt) * derive.ddy_bwd((v + 0.5 * k1_v * dt), dy) + nu * derive.laplacian(
        (v + 0.5 * k1_v * dt), dx, dy))[:, 1]
    # periodic condition at x=0
    utemp = np.hstack(
        ((u + 0.5 * k1_u * dt)[:, -1].reshape((ny, 1)), (u + 0.5 * k1_u * dt)[:, :2].reshape((ny, 2))))
    vtemp = np.hstack(
        ((v + 0.5 * k1_v * dt)[:, -1].reshape((ny, 1)), (v + 0.5 * k1_v * dt)[:, :2].reshape((ny, 2))))
    k2_u[:, 0] = (- (u + 0.5 * k1_u * dt) * derive.ddx_bwd((u + 0.5 * k1_u * dt), dx) - (
            v + 0.5 * k1_v * dt) * derive.ddy_bwd((u + 0.5 * k1_u * dt), dy) + nu * derive.laplacian(
        (u + 0.5 * k1_u * dt), dx, dy) - dpdx)[:, 1]
    k2_v[:, 0] = (- (u + 0.5 * k1_u * dt) * derive.ddx_bwd((v + 0.5 * k1_v * dt), dx) - (
            v + 0.5 * k1_v * dt) * derive.ddy_bwd((v + 0.5 * k1_v * dt), dy) + nu * derive.laplacian(
        (v + 0.5 * k1_v * dt), dx, dy) - dpdy)[:, 1]

    t = t + 0.5 * dt
    k3_u = - (u - k1_u * dt + 2 * k2_u * dt) * derive.ddx_bwd((u - k1_u * dt + 2 * k2_u * dt), dx) - (
            v - k1_v * dt + 2 * k2_v * dt) * derive.ddy_bwd((u - k1_u * dt + 2 * k2_u * dt),
                                                            dy) + nu * derive.laplacian(
        (u - k1_u * dt + 2 * k2_u * dt), dx, dy)
    k3_v = - (u - k1_u * dt + 2 * k2_u * dt) * derive.ddx_bwd((v - k1_v * dt + 2 * k2_v * dt), dx) - (
            v - k1_v * dt + 2 * k2_v * dt) * derive.ddy_bwd((v - k1_v * dt + 2 * k2_v * dt),
                                                            dy) + nu * derive.laplacian(
        (v - k1_v * dt + 2 * k2_v * dt), dx, dy)

    # periodic condition at x=lx
    utemp = np.hstack(((u - k1_u * dt + 2 * k2_u * dt)[:, -2:].reshape((ny, 2)),
                       (u - k1_u * dt + 2 * k2_u * dt)[:, 0].reshape((ny, 1))))
    vtemp = np.hstack(((v - k1_v * dt + 2 * k2_v * dt)[:, -2:].reshape((ny, 2)),
                       (v - k1_v * dt + 2 * k2_v * dt)[:, 0].reshape((ny, 1))))
    k3_u[:, -1] = (- (u - k1_u * dt + 2 * k2_u * dt) * derive.ddx_bwd((u - k1_u * dt + 2 * k2_u * dt),
                                                                      dx) - (
                           v - k1_v * dt + 2 * k2_v * dt) * derive.ddy_bwd(
        (u - k1_u * dt + 2 * k2_u * dt), dy) + nu * derive.laplacian((u - k1_u * dt + 2 * k2_u * dt), dx, dy))[:,
                  1]
    k3_v[:, -1] = (- (u - k1_u * dt + 2 * k2_u * dt) * derive.ddx_bwd((v - k1_v * dt + 2 * k2_v * dt),
                                                                      dx) - (
                           v - k1_v * dt + 2 * k2_v * dt) * derive.ddy_bwd(
        (v - k1_v * dt + 2 * k2_v * dt), dy) + nu * derive.laplacian((v - k1_v * dt + 2 * k2_v * dt), dx, dy))[:,
                  1]
    # periodic condition at x=0
    utemp = np.hstack(((u - k1_u * dt + 2 * k2_u * dt)[:, -1].reshape((ny, 1)),
                       (u - k1_u * dt + 2 * k2_u * dt)[:, :2].reshape((ny, 2))))
    vtemp = np.hstack(((v - k1_v * dt + 2 * k2_v * dt)[:, -1].reshape((ny, 1)),
                       (v - k1_v * dt + 2 * k2_v * dt)[:, :2].reshape((ny, 2))))
    k3_u[:, 0] = (- (u - k1_u * dt + 2 * k2_u * dt) * derive.ddx_bwd((u - k1_u * dt + 2 * k2_u * dt),
                                                                     dx) - (
                          v - k1_v * dt + 2 * k2_v * dt) * derive.ddy_bwd(
        (u - k1_u * dt + 2 * k2_u * dt), dy) + nu * derive.laplacian((u - k1_u * dt + 2 * k2_u * dt), dx,
                                                                     dy) - dpdx)[:, 1]
    k3_v[:, 0] = (- (u - k1_u * dt + 2 * k2_u * dt) * derive.ddx_bwd((v - k1_v * dt + 2 * k2_v * dt),
                                                                     dx) - (
                          v - k1_v * dt + 2 * k2_v * dt) * derive.ddy_bwd(
        (v - k1_v * dt + 2 * k2_v * dt), dy) + nu * derive.laplacian((v - k1_v * dt + 2 * k2_v * dt), dx,
                                                                     dy) - dpdy)[:, 1]

    uRHS_conv_diff = 1 / 6 * (k1_u + 4 * k2_u + k3_u)

    vRHS_conv_diff = 1 / 6 * (k1_v + 4 * k2_v + k3_v)

    uRHS = uRHS_conv_diff - dpdx

    vRHS = vRHS_conv_diff - dpdy

    ibm_forcing_u, ibm_forcing_v=ibm.circle(uRHS, vRHS, u, v, dt, x, y, xx, yy, r, R, theta, nx, ny, F, epsilon)

    ustar = u + dt * uRHS + F * dt + ibm_forcing_u * dt
    vstar = v + dt * vRHS + ibm_forcing_v * dt

    ustar[0, :] = 0
    ustar[-1, :] = 0
    vstar[0, :] = 0
    vstar[-1, :] = 0

    # Step2
    ustarstar, vstarstar = projection_method.step2(ustar, vstar, dpdx, dpdy, dt)

    # Step3
    p = projection_method.step3(ustarstar, vstarstar, rho, epsilon, dx, dy, nx, ny, nx_sp, ny_sp, K, dt)

    # Step4
    u, v, dpdx, dpdy = projection_method.step4(ustarstar, vstarstar, p, dx, dy, nx, ny, dt)

    if np.mod(stepcount, saveth_iter) == 0:
        ip_op.write_szl_2D(xx, yy, p, u, v, stepcount * dt, int(stepcount / saveth_iter))

    print(stepcount)

    return u, v, dpdx, dpdy, uRHS_conv_diff, vRHS_conv_diff
