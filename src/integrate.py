import numpy as np
from src import derive, projection_method, ibm, ip_op


def rk3(u, v, nx, ny, nu, dx, dy, dt, dpdx, dpdy, epsilon, F, theta, r, R, rho, stepcount,
        saveth_iter, x, y, xx, yy, nx_sp, ny_sp, K, bc):
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0

    # Step1 with RK3
    t = 0
    k1_u = derive.conv_diff_u(u, v, nu, dx, dy, bc)
    k1_v = derive.conv_diff_v(u, v, nu, dx, dy, bc)

    t = t + 0.5 * dt
    k2_u = derive.conv_diff_u((u + 0.5 * k1_u * dt), (v + 0.5 * k1_v * dt), nu, dx, dy, bc)
    k2_v = derive.conv_diff_v((u + 0.5 * k1_u * dt), (v + 0.5 * k1_v * dt), nu, dx, dy, bc)

    t = t + 0.5 * dt
    k3_u = derive.conv_diff_u((u - k1_u * dt + 2 * k2_u * dt), (v - k1_v * dt + 2 * k2_v * dt), nu, dx, dy, bc)
    k3_v = derive.conv_diff_v((u - k1_u * dt + 2 * k2_u * dt), (v - k1_v * dt + 2 * k2_v * dt), nu, dx, dy, bc)

    uRHS_conv_diff = 1 / 6 * (k1_u + 4 * k2_u + k3_u)
    vRHS_conv_diff = 1 / 6 * (k1_v + 4 * k2_v + k3_v)

    uRHS = uRHS_conv_diff - dpdx
    vRHS = vRHS_conv_diff - dpdy

    ibm_forcing_u, ibm_forcing_v = ibm.circle(uRHS, vRHS, u, v, dt, x, y, xx, yy, r, R, theta, nx, ny, F, epsilon)

    ustar = u + dt * uRHS + F * dt + ibm_forcing_u * dt
    vstar = v + dt * vRHS + ibm_forcing_v * dt

    ustar[0, :] = 0
    ustar[-1, :] = 0
    vstar[0, :] = 0
    vstar[-1, :] = 0

    # Step2
    ustarstar, vstarstar = projection_method.step2(ustar, vstar, dpdx, dpdy, dt)

    # Step3
    p = projection_method.step3(ustarstar, vstarstar, rho, epsilon, dx, dy, nx_sp, ny_sp, K, dt, bc)

    # Step4
    u, v, dpdx, dpdy = projection_method.step4(ustarstar, vstarstar, p, dx, dy, dt, bc)

    if np.mod(stepcount, saveth_iter) == 0:
        ip_op.write_szl_2D(xx, yy, p, u, v, stepcount * dt, int(stepcount / saveth_iter))

    print(stepcount)

    return u, v, dpdx, dpdy, uRHS_conv_diff, vRHS_conv_diff
