def PyDNS():
    import numpy as np
    from matplotlib import pyplot
    from src import pressure_poisson
    from src import diff_ops
    from src import ip_op

    ##variable declarations
    nx = 41
    ny = 41
    nit = 50
    lx = 2
    ly = 2
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    xx, yy = np.meshgrid(x, y)
    nt = 500
    saveth_iter = 25

    ##physical variables
    rho = 1
    nu = .1
    F = 1
    dt = .01

    # initial conditions
    u = np.zeros((ny, nx))
    utemp = np.zeros((ny, 3))

    v = np.zeros((ny, nx))
    vtemp = np.zeros((ny, 3))

    p = np.zeros((ny, nx))
    ptemp = np.zeros((ny, 3))
    dpdx = np.zeros((ny, nx))
    dpdy = np.zeros((ny, nx))

    ip_op.write_szl_2D(xx, yy, p, u, v, 0, 0)

    for stepcount in range(1, nt + 1):
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        # Step1
        # do the x-momentum RHS
        # u rhs: - d(uu)/dx - d(vu)/dy + ν d2(u)
        uRHS = - diff_ops.ddx(u * u, dx) - diff_ops.ddy(v * u, dy) + nu * diff_ops.laplacian(u, dx, dy) - dpdx
        # v rhs: - d(uv)/dx - d(vv)/dy + ν d2(v)
        vRHS = - diff_ops.ddx(u * v, dx) - diff_ops.ddy(v * v, dy) + nu * diff_ops.laplacian(v, dx, dy) - dpdx

        # periodic condition at x=lx
        utemp = np.hstack((u[:, -2:].reshape((ny, 2)), u[:, 0].reshape((ny, 1))))
        vtemp = np.hstack((v[:, -2:].reshape((ny, 2)), v[:, 0].reshape((ny, 1))))
        uRHS[:, -1] = (- diff_ops.ddx(utemp * utemp, dx) - diff_ops.ddy(vtemp * utemp, dy) + nu * diff_ops.laplacian(
            utemp, dx, dy))[:, 1]
        vRHS[:, -1] = (- diff_ops.ddx(utemp * vtemp, dx) - diff_ops.ddy(vtemp * vtemp, dy) + nu * diff_ops.laplacian(
            vtemp, dx, dy))[:, 1]

        # periodic condition at x=0
        utemp = np.hstack((u[:, -1].reshape((ny, 1)), u[:, :2].reshape((ny, 2))))
        vtemp = np.hstack((v[:, -1].reshape((ny, 1)), v[:, :2].reshape((ny, 2))))
        uRHS[:, 0] = (- diff_ops.ddx(utemp * utemp, dx) - diff_ops.ddy(vtemp * utemp, dy) + nu * diff_ops.laplacian(
            utemp, dx, dy))[:, 1]
        vRHS[:, 0] = (- diff_ops.ddx(utemp * vtemp, dx) - diff_ops.ddy(vtemp * vtemp, dy) + nu * diff_ops.laplacian(
            vtemp, dx, dy))[:, 1]

        ustar = u + dt * uRHS + F * dt
        vstar = v + dt * vRHS

        ustar[0, :] = 0
        ustar[-1, :] = 0
        vstar[0, :] = 0
        vstar[-1, :] = 0

        # Step2
        ustarstar = ustar + dpdx * dt
        vstarstar = vstar + dpdy * dt

        # Step3
        # next compute the pressure RHS: prhs = div(un)/dt + div( [urhs, vrhs])
        prhs = rho * diff_ops.div(ustarstar, vstarstar, dx, dy) / dt

        # periodic condition at x=lx
        utemp = np.hstack((ustarstar[:, -2:].reshape((ny, 2)), ustarstar[:, 0].reshape((ny, 1))))
        vtemp = np.hstack((vstarstar[:, -2:].reshape((ny, 2)), vstarstar[:, 0].reshape((ny, 1))))
        prhs[:, -1] = (rho * diff_ops.div(utemp, vtemp, dx, dy) / dt)[:, 1]

        # periodic condition at x=0
        utemp = np.hstack((ustarstar[:, -1].reshape((ny, 1)), ustarstar[:, :2].reshape((ny, 2))))
        vtemp = np.hstack((vstarstar[:, -1].reshape((ny, 1)), vstarstar[:, :2].reshape((ny, 2))))
        prhs[:, 0] = (rho * diff_ops.div(utemp, vtemp, dx, dy) / dt)[:, 1]

        p, err = pressure_poisson.solve_new(p, dx, dy, prhs)

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

        dpdy = diff_ops.ddx(p, dy)

        u = ustarstar - dt * dpdx
        v = vstarstar - dt * dpdy

        if np.mod(stepcount, saveth_iter) == 0:
            ip_op.write_szl_2D(xx, yy, p, u, v, stepcount * dt, int(stepcount / saveth_iter))

        print(stepcount)

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(xx[::3, ::3], yy[::3, ::3], u[::3, ::3], v[::3, ::3])

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(xx, yy, u, v)

    pyplot.show()


if __name__ == "__main__":
    import os

    if not os.path.exists('data'):
        os.makedirs('data')

    PyDNS()
