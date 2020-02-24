import numpy as np
from src import derive, pressure_poisson, ibm


def step1(u, v, nx, ny, nu, x, y, xx, yy, dx, dy, dt, epsilon, F, R, theta, r, uRHS_conv_diff_p, uRHS_conv_diff_pp,
          vRHS_conv_diff_p, vRHS_conv_diff_pp, dpdx, dpdy, bc):

    # Step1
    # do the x-momentum RHS
    # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
    uRHS_conv_diff = derive.conv_diff_u(u, v, nu, dx, dy, bc)
    # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
    vRHS_conv_diff = derive.conv_diff_v(u, v, nu, dx, dy, bc)

    uRHS = (23 * uRHS_conv_diff - 16 * uRHS_conv_diff_p + 5 * uRHS_conv_diff_pp) / 12 - dpdx

    vRHS = (23 * vRHS_conv_diff - 16 * vRHS_conv_diff_p + 5 * vRHS_conv_diff_pp) / 12 - dpdy

    ibm_forcing_u, ibm_forcing_v = ibm.circle(uRHS, vRHS, u, v, dt, x, y, xx, yy, r, R, theta, nx, ny, F, epsilon)

    ustar = u + dt * uRHS + F * dt + ibm_forcing_u * dt
    vstar = v + dt * vRHS + ibm_forcing_v * dt

    if bc['y']=='no-slip':
        ustar[0, :] = 0
        ustar[-1, :] = 0
        vstar[0, :] = 0
        vstar[-1, :] = 0

    if bc['x']=='dirichlet':
        ustar[:, -1] = u[:, -1]-1*(u[:, -1]-u[:, -2])/dx


    return ustar, vstar, uRHS_conv_diff, vRHS_conv_diff


def step2(ustar, vstar, dpdx, dpdy, dt):
    ustarstar = ustar + dpdx * dt
    vstarstar = vstar + dpdy * dt
    return ustarstar, vstarstar


def step3(ustarstar, vstarstar, rho, epsilon, dx, dy, nx_sp, ny_sp, K, dt, bc):
    # next compute the pressure RHS: prhs = div(un)/dt + div( [urhs, vrhs])
    prhs = rho * derive.div((1 - epsilon) * ustarstar, (1 - epsilon) * vstarstar, dx, dy, bc) / dt

    # p, err = pressure_poisson.solve_new(p, dx, dy, prhs)

    p = pressure_poisson.solve_spectral(nx_sp, ny_sp, K, prhs)

    return p


def step4(ustarstar, vstarstar, p, dx, dy, dt, bc):
    # Step4
    # finally compute the true velocities
    # u_{n+1} = uh - dt*dpdx
    dpdx = derive.ddx(p, dx, bc)

    dpdy = derive.ddy(p, dy, bc)

    u = ustarstar - dt * dpdx
    v = vstarstar - dt * dpdy

    return u, v, dpdx, dpdy
