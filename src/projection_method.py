import numpy as np
from src import derive, pressure_poisson, ibm
from scipy import interpolate


def step1(u, v, nx, ny, nu, x, y, xx, yy, dx, dy, dt, epsilon, F, R, theta, r, uRHS_conv_diff_p, uRHS_conv_diff_pp,
          vRHS_conv_diff_p, vRHS_conv_diff_pp, dpdx, dpdy):
    u_desired = np.zeros_like(u)
    v_desired = np.zeros_like(v)
    # Step1
    # do the x-momentum RHS
    # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
    uRHS_conv_diff = - u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(u, dx, dy)
    # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
    vRHS_conv_diff = - u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(v, dx, dy)

    # periodic condition at x=lx
    utemp = np.hstack((u[:, -2:].reshape((ny, 2)), u[:, 0].reshape((ny, 1))))
    vtemp = np.hstack((v[:, -2:].reshape((ny, 2)), v[:, 0].reshape((ny, 1))))
    uRHS_conv_diff[:, -1] = (- u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(u, dx,
                                                                                                             dy))[:, 1]
    vRHS_conv_diff[:, -1] = (- u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(v, dx,
                                                                                                             dy))[:, 1]
    # periodic condition at x=0
    utemp = np.hstack((u[:, -1].reshape((ny, 1)), u[:, :2].reshape((ny, 2))))
    vtemp = np.hstack((v[:, -1].reshape((ny, 1)), v[:, :2].reshape((ny, 2))))
    uRHS_conv_diff[:, 0] = (- u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(u, dx,
                                                                                                            dy) - dpdx)[
                           :, 1]
    vRHS_conv_diff[:, 0] = (- u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(v, dx,
                                                                                                            dy) - dpdy)[
                           :, 1]

    uRHS = (23 * uRHS_conv_diff - 16 * uRHS_conv_diff_p + 5 * uRHS_conv_diff_pp) / 12 - dpdx

    vRHS = (23 * vRHS_conv_diff - 16 * vRHS_conv_diff_p + 5 * vRHS_conv_diff_pp) / 12 - dpdy

    ibm_forcing_u, ibm_forcing_v=ibm.circle(uRHS, vRHS, u, v, dt, x, y, xx, yy, r, R, theta, nx, ny, F, epsilon)

    ustar = u + dt * uRHS + F * dt + ibm_forcing_u * dt
    vstar = v + dt * vRHS + ibm_forcing_v * dt

    ustar[0, :] = 0
    ustar[-1, :] = 0
    vstar[0, :] = 0
    vstar[-1, :] = 0

    return ustar, vstar, uRHS_conv_diff, vRHS_conv_diff


def step2(ustar, vstar, dpdx, dpdy, dt):
    ustarstar = ustar + dpdx * dt
    vstarstar = vstar + dpdy * dt
    return ustarstar, vstarstar


def step3(ustarstar, vstarstar, rho, epsilon, dx, dy, nx, ny, nx_sp, ny_sp, K, dt):
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

    return p


def step4(ustarstar, vstarstar, p, dx, dy, nx, ny, dt):
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

    return u, v, dpdx, dpdy
