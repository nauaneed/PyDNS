def PyDNS():
    import numpy as np
    from matplotlib import pyplot
    from src import pressure_poisson

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
    nt = 499

    ##physical variables
    rho = 1
    nu = .1
    F = 1
    dt = .01

    # initial conditions
    u = np.zeros((ny, nx))
    un = np.zeros((ny, nx))

    v = np.zeros((ny, nx))
    vn = np.zeros((ny, nx))

    p = np.ones((ny, nx))
    pn = np.ones((ny, nx))

    b = np.zeros((ny, nx))

    udiff = 1
    stepcount = 0

    while udiff > .001:
        un = u.copy()
        vn = v.copy()

        p, err = pressure_poisson.solve_new(p, dx, dy, pressure_poisson.build_up_b(rho, dt, dx, dy, u, v))

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) *
                         (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx ** 2 *
                               (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy ** 2 *
                               (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                         F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) *
                         (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx ** 2 *
                               (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy ** 2 *
                               (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2
        u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx *
                       (un[1:-1, -1] - un[1:-1, -2]) -
                       vn[1:-1, -1] * dt / dy *
                       (un[1:-1, -1] - un[0:-2, -1]) -
                       dt / (2 * rho * dx) *
                       (p[1:-1, 0] - p[1:-1, -2]) +
                       nu * (dt / dx ** 2 *
                             (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2]) +
                             dt / dy ** 2 *
                             (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                      (un[1:-1, 0] - un[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy *
                      (un[1:-1, 0] - un[0:-2, 0]) -
                      dt / (2 * rho * dx) *
                      (p[1:-1, 1] - p[1:-1, -1]) +
                      nu * (dt / dx ** 2 *
                            (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                            dt / dy ** 2 *
                            (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                       (vn[1:-1, -1] - vn[1:-1, -2]) -
                       vn[1:-1, -1] * dt / dy *
                       (vn[1:-1, -1] - vn[0:-2, -1]) -
                       dt / (2 * rho * dy) *
                       (p[2:, -1] - p[0:-2, -1]) +
                       nu * (dt / dx ** 2 *
                             (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                             dt / dy ** 2 *
                             (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                      (vn[1:-1, 0] - vn[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy *
                      (vn[1:-1, 0] - vn[0:-2, 0]) -
                      dt / (2 * rho * dy) *
                      (p[2:, 0] - p[0:-2, 0]) +
                      nu * (dt / dx ** 2 *
                            (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                            dt / dy ** 2 *
                            (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1

        print(stepcount)

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(xx[::3, ::3], yy[::3, ::3], u[::3, ::3], v[::3, ::3])

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(xx, yy, u, v)

    pyplot.show()


if __name__ == "__main__":
    PyDNS()
