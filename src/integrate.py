import numpy as np
from src import pressure_poisson
from src import derive
from src import ip_op
from scipy import interpolate


def rk3(u, v, nx, ny, nu, dx, dy, dt, dpdx, dpdy, epsilon, F, u_desired, v_desired, theta, r, R, rho, stepcount,
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

    interpolate_u = interpolate.interp2d(x, y, u + uRHS * dt + F * dt, kind='cubic')
    interpolate_v = interpolate.interp2d(x, y, v + vRHS * dt, kind='cubic')

    for i in range(nx):
        for j in range(ny):
            if epsilon[j, i] != 0:
                u_desired[j, i] = -np.sin(2 * np.pi * (r[j, i] ** 2) / (2 * (R ** 2))) * interpolate_u(
                    xx[j, i] + (R - r[j, i]) * np.cos(theta[j, i]),
                    yy[j, i] + (R - r[j, i]) * np.sin(theta[j, i]))
                v_desired[j, i] = -np.sin(2 * np.pi * (r[j, i] ** 2) / (2 * (R ** 2))) * interpolate_v(
                    xx[j, i] + (R - r[j, i]) * np.cos(theta[j, i]),
                    yy[j, i] + (R - r[j, i]) * np.sin(theta[j, i]))

    ibm_forcing_u = epsilon * (-uRHS - F + (u_desired - u) / dt)
    ibm_forcing_v = epsilon * (-vRHS + (v_desired - v) / dt)

    ustar = u + dt * uRHS + F * dt + ibm_forcing_u * dt
    vstar = v + dt * vRHS + ibm_forcing_v * dt

    ustar[0, :] = 0
    ustar[-1, :] = 0
    vstar[0, :] = 0
    vstar[-1, :] = 0

    # Step2
    ustarstar = ustar + dpdx * dt
    vstarstar = vstar + dpdy * dt

    # Step3
    # next compute the pressure RHS: prhs = div(un)/dt + div( [urhs, vrhs])
    prhs = rho * derive.div((1 - epsilon) * ustarstar, (1 - epsilon) * vstarstar, dx, dy) / dt

    # periodic condition at x=lx
    utemp = np.hstack((ustarstar[:, -2:].reshape((ny, 2)), ustarstar[:, 0].reshape((ny, 1))))
    vtemp = np.hstack((vstarstar[:, -2:].reshape((ny, 2)), vstarstar[:, 0].reshape((ny, 1))))
    prhs[:, -1] = (rho * derive.div(utemp, vtemp, dx, dy) / dt)[:, 1]

    # periodic condition at x=0
    utemp = np.hstack((ustarstar[:, -1].reshape((ny, 1)), ustarstar[:, :2].reshape((ny, 2))))
    vtemp = np.hstack((vstarstar[:, -1].reshape((ny, 1)), vstarstar[:, :2].reshape((ny, 2))))
    prhs[:, 0] = (rho * derive.div(utemp, vtemp, dx, dy) / dt)[:, 1]

    # p, err = pressure_poisson.solve_new(p, dx, dy, prhs)

    p = pressure_poisson.solve_spectral(nx_sp, ny_sp, K, prhs)

    # Step4
    # finally compute the true velocities
    # u_{n+1} = uh - dt*dpdx
    dpdx = derive.ddx(p, dx)

    # periodic condition at x=lx
    ptemp = np.hstack((p[:, -2:].reshape((ny, 2)), p[:, 0].reshape((ny, 1))))
    dpdx[:, -1] = derive.ddx(ptemp, dx)[:, 1]

    # periodic condition at x=0
    ptemp = np.hstack((p[:, -1].reshape((ny, 1)), p[:, :2].reshape((ny, 2))))
    dpdx[:, 0] = derive.ddx(ptemp, dx)[:, 1]

    dpdy = derive.ddy(p, dy)

    u = ustarstar - dt * dpdx
    v = vstarstar - dt * dpdy

    if np.mod(stepcount, saveth_iter) == 0:
        ip_op.write_szl_2D(xx, yy, p, u, v, stepcount * dt, int(stepcount / saveth_iter))

    print(stepcount)

    return u, v, dpdx, dpdy, uRHS_conv_diff, vRHS_conv_diff
