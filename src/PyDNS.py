def PyDNS():
    import numpy as np
    from matplotlib import pyplot
    from src import pressure_poisson
    from src import diff_ops
    from src import ip_op
    from scipy import interpolate

    ##variable declarations
    nx = 128
    ny = 64
    lx = 40
    ly = 20
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xx, yy = np.meshgrid(x, y)
    nt = 100000
    saveth_iter = 500

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

    #ibm
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

    for stepcount in range(1, nt + 1):
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        # Step1
        # do the x-momentum RHS
        # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
        uRHS = - u*diff_ops.ddx_bwd(u, dx) - v*diff_ops.ddy_bwd(u, dy) + nu * diff_ops.laplacian(u, dx, dy) - dpdx
        # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
        vRHS = - u*diff_ops.ddx_bwd(v, dx) - v*diff_ops.ddy_bwd(v, dy) + nu * diff_ops.laplacian(v, dx, dy) - dpdy

        # periodic condition at x=lx
        utemp = np.hstack((u[:, -2:].reshape((ny, 2)), u[:, 0].reshape((ny, 1))))
        vtemp = np.hstack((v[:, -2:].reshape((ny, 2)), v[:, 0].reshape((ny, 1))))
        uRHS[:, -1] = (- u*diff_ops.ddx_bwd(u, dx) - v*diff_ops.ddy_bwd(u, dy) + nu * diff_ops.laplacian(u, dx, dy) - dpdx)[:,1]

        vRHS[:, -1] = (- u*diff_ops.ddx_bwd(v, dx) - v*diff_ops.ddy_bwd(v, dy) + nu * diff_ops.laplacian(v, dx, dy) - dpdy)[:,1]
        # periodic condition at x=0
        utemp = np.hstack((u[:, -1].reshape((ny, 1)), u[:, :2].reshape((ny, 2))))
        vtemp = np.hstack((v[:, -1].reshape((ny, 1)), v[:, :2].reshape((ny, 2))))
        uRHS[:, 0] = (- u*diff_ops.ddx_bwd(u, dx) - v*diff_ops.ddy_bwd(u, dy) + nu * diff_ops.laplacian(u, dx, dy) - dpdx)[:,1]
        vRHS[:, 0] = (- u*diff_ops.ddx_bwd(v, dx) - v*diff_ops.ddy_bwd(v, dy) + nu * diff_ops.laplacian(v, dx, dy) - dpdy)[:,1]
        interpolate_u = interpolate.interp2d(x, y, u+uRHS*dt+F*dt, kind='cubic')
        interpolate_v = interpolate.interp2d(x, y, v+vRHS*dt, kind='cubic')

        for i in range(nx):
            for j in range(ny):
                if epsilon[j, i] != 0:
                    u_desired[j, i] = -np.sin(2 * np.pi * (r[j, i] ** 2) / (2 * (R ** 2))) * interpolate_u(
                        xx[j, i] + (R - r[j, i]) * np.cos(theta[j, i]),
                        yy[j, i] + (R - r[j, i]) * np.sin(theta[j, i]))
                    v_desired[j, i] = -np.sin(2 * np.pi * (r[j, i] ** 2) / (2 * (R ** 2))) * interpolate_v(
                        xx[j, i] + (R - r[j, i]) * np.cos(theta[j, i]),
                        yy[j, i] + (R - r[j, i]) * np.sin(theta[j, i]))

        ibm_forcing_u = epsilon * (-uRHS -F + (u_desired - u) / dt)
        ibm_forcing_v = epsilon * (-vRHS + (v_desired - v) / dt)

        ustar = u + dt * uRHS + F * dt+ibm_forcing_u*dt
        vstar = v + dt * vRHS+ibm_forcing_v*dt

        ustar[0, :] = 0
        ustar[-1, :] = 0
        vstar[0, :] = 0
        vstar[-1, :] = 0

        # Step2
        ustarstar = ustar + dpdx * dt
        vstarstar = vstar + dpdy * dt

        # Step3
        # next compute the pressure RHS: prhs = div(un)/dt + div( [urhs, vrhs])
        prhs = rho * diff_ops.div((1-epsilon)*ustarstar, (1-epsilon)*vstarstar, dx, dy) / dt

        # periodic condition at x=lx
        utemp = np.hstack((ustarstar[:, -2:].reshape((ny, 2)), ustarstar[:, 0].reshape((ny, 1))))
        vtemp = np.hstack((vstarstar[:, -2:].reshape((ny, 2)), vstarstar[:, 0].reshape((ny, 1))))
        prhs[:, -1] = (rho * diff_ops.div(utemp, vtemp, dx, dy) / dt)[:, 1]

        # periodic condition at x=0
        utemp = np.hstack((ustarstar[:, -1].reshape((ny, 1)), ustarstar[:, :2].reshape((ny, 2))))
        vtemp = np.hstack((vstarstar[:, -1].reshape((ny, 1)), vstarstar[:, :2].reshape((ny, 2))))
        prhs[:, 0] = (rho * diff_ops.div(utemp, vtemp, dx, dy) / dt)[:, 1]

        #p, err = pressure_poisson.solve_new(p, dx, dy, prhs)

        p = pressure_poisson.solve_spectral(nx_sp, ny_sp, K, prhs)

        # Step4
        # finally compute the true velocities
        # u_{n+1} = uh - dt*dpdx
        dpdx = diff_ops.ddx(p, dx)

        # periodic condition at x=lx
        ptemp = np.hstack((p[:, -2:].reshape((ny, 2)), p[:, 0].reshape((ny, 1))))
        dpdx[:, -1] = diff_ops.ddx(ptemp, dx)[:, 1]

        # periodic condition at x=0
        ptemp = np.hstack((p[:, -1].reshape((ny, 1)), p[:, :2].reshape((ny, 2))))
        dpdx[:, 0] = diff_ops.ddx(ptemp, dx)[:, 1]

        dpdy = diff_ops.ddy(p, dy)

        u = ustarstar - dt * dpdx
        v = vstarstar - dt * dpdy

        if np.mod(stepcount, saveth_iter) == 0:
            ip_op.write_szl_2D(xx, yy, p, u, v, stepcount * dt, int(stepcount / saveth_iter))

        print(stepcount)

    '''
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(xx[::3, ::3], yy[::3, ::3], u[::3, ::3], v[::3, ::3])

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(xx, yy, u, v)

    pyplot.show()
    '''

if __name__ == "__main__":
    import os

    if not os.path.exists('data'):
        os.makedirs('data')

    PyDNS()
