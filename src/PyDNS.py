def PyDNS():
    import numpy as np
    from matplotlib import pyplot
    from src import pressure_poisson, integrate
    from src import derive
    from src import ip_op
    from scipy import interpolate

    ##variable declarations
    nx = 256
    ny = 128
    lx = 40
    ly = 20
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xx, yy = np.meshgrid(x, y)
    nt = 1000000
    saveth_iter = 5000

    ##physical variables
    rho = 1
    nu = .5
    F = 0.5
    dt = .001

    # initial conditions
    u = np.ones((ny, nx))
    utemp = np.zeros((ny, 3))

    v = np.zeros((ny, nx))
    vtemp = np.zeros((ny, 3))

    p = np.zeros((ny, nx))
    ptemp = np.zeros((ny, 3))
    dpdx = np.zeros((ny, nx))
    dpdy = np.zeros((ny, nx))
    epsilon = np.zeros((ny, nx))
    u_desired = np.zeros((ny, nx))
    v_desired = np.zeros((ny, nx))
    uRHS_conv_diff_p = np.zeros((ny, nx))
    uRHS_conv_diff_pp = np.zeros((ny, nx))
    vRHS_conv_diff_p = np.zeros((ny, nx))
    vRHS_conv_diff_pp = np.zeros((ny, nx))

    # ibm
    r = ((xx - lx / 4) ** 2 + (yy - ly / 2) ** 2) ** 0.5
    theta = np.arctan2(yy - ly / 2, xx - lx / 4)

    R = 1

    for i in range(nx):
        for j in range(ny):
            if r[j, i] <= R:
                epsilon[j, i] = 1

    ## pressure_poisson
    nx_sp = nx
    ny_sp = ny

    kx = np.array([(2 * np.pi * i / lx) for i in range(0, (int(nx_sp / 2) - 1))])
    kx = np.append(kx, np.array([(2 * np.pi * (nx_sp - i) / lx) for i in range(int(nx_sp / 2) - 1, nx_sp)]))
    ky = np.array([(np.pi * (i + 1) / ly) for i in range(0, ny_sp)])
    KX, KY = np.meshgrid(kx, ky)
    K = KX ** 2 + KY ** 2

    ip_op.write_szl_2D(xx, yy, p, u, v, 0, 0)

    ## rk3

    for stepcount in range(1, 3):
        u, v, dpdx, dpdy, uRHS_conv_diff, vRHS_conv_diff = integrate.rk3(u, v, nx, ny, nu, dx, dy, dt, dpdx, dpdy,
                                                                         epsilon, F, u_desired, v_desired, theta, r, R,
                                                                         rho, stepcount, saveth_iter, x, y, xx, yy,
                                                                         nx_sp, ny_sp, K)

        uRHS_conv_diff_pp = uRHS_conv_diff_p.copy()
        vRHS_conv_diff_pp = vRHS_conv_diff_p.copy()

        uRHS_conv_diff_p = uRHS_conv_diff.copy()
        vRHS_conv_diff_p = vRHS_conv_diff.copy()

    for stepcount in range(3, nt + 1):
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        # Step1
        # do the x-momentum RHS
        # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
        uRHS_conv_diff = - u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(u, dx,
                                                                                                         dy)
        # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
        vRHS_conv_diff = - u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(v, dx,
                                                                                                         dy)

        # periodic condition at x=lx
        utemp = np.hstack((u[:, -2:].reshape((ny, 2)), u[:, 0].reshape((ny, 1))))
        vtemp = np.hstack((v[:, -2:].reshape((ny, 2)), v[:, 0].reshape((ny, 1))))
        uRHS_conv_diff[:, -1] = (- u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy) + nu * derive.laplacian(
            u, dx, dy))[:, 1]
        vRHS_conv_diff[:, -1] = (- u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy) + nu * derive.laplacian(
            v, dx, dy))[:, 1]
        # periodic condition at x=0
        utemp = np.hstack((u[:, -1].reshape((ny, 1)), u[:, :2].reshape((ny, 2))))
        vtemp = np.hstack((v[:, -1].reshape((ny, 1)), v[:, :2].reshape((ny, 2))))
        uRHS_conv_diff[:, 0] = (- u * derive.ddx_bwd(u, dx) - v * derive.ddy_bwd(u, dy)
                                + nu * derive.laplacian(u, dx, dy) - dpdx)[:, 1]
        vRHS_conv_diff[:, 0] = (- u * derive.ddx_bwd(v, dx) - v * derive.ddy_bwd(v, dy)
                                + nu * derive.laplacian(v, dx, dy) - dpdy)[:, 1]

        uRHS = (23 * uRHS_conv_diff - 16 * uRHS_conv_diff_p + 5 * uRHS_conv_diff_pp) / 12 - dpdx

        vRHS = (23 * vRHS_conv_diff - 16 * vRHS_conv_diff_p + 5 * vRHS_conv_diff_pp) / 12 - dpdy

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


        fig = pyplot.figure(figsize=(11, 7), dpi=100)
        pyplot.quiver(xx[::3, ::3], yy[::3, ::3], u[::3, ::3], v[::3, ::3])
        '''
        fig = pyplot.figure(figsize=(11, 7), dpi=100)
        pyplot.quiver(xx, yy, u, v)
        '''
        pyplot.show()



if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    from urllib import request

    # To download tecio library module
    lib_folder = Path("tecio")
    if os.name == 'nt':
        dll_path = lib_folder / 'libtecio.dll'
        if not dll_path.is_file():
            url = 'https://raw.githubusercontent.com/blacksong/pytecio/master/2017r3_tecio.dll'
            print('Downloading dll from github:', url)
            request.urlretrieve(url, dll_path)

    else:
        dll_path = lib_folder / 'libtecio.so'
        if not dll_path.is_file():
            url = 'https://raw.githubusercontent.com/blacksong/pytecio/master/2017r2_tecio.so'
            print('Downloading dll from github:', url)
            request.urlretrieve(url, dll_path)

    if not os.path.exists('data'):
        os.makedirs('data')

    PyDNS()
