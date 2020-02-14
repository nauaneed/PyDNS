def PyDNS():
    import numpy as np
    from matplotlib import pyplot
    from src import integrate, projection_method, ip_op

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
    nt = 300
    saveth_iter = 20

    ##physical variables
    rho = 1
    nu = .5
    F = 0.5
    dt = .001

    # boundary conditions
    bc = {'x': 'wall', 'y': 'periodic'}

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
                                                                         epsilon, F, theta, r, R,
                                                                         rho, stepcount, saveth_iter, x, y, xx, yy,
                                                                         nx_sp, ny_sp, K)

        uRHS_conv_diff_pp = uRHS_conv_diff_p.copy()
        vRHS_conv_diff_pp = vRHS_conv_diff_p.copy()

        uRHS_conv_diff_p = uRHS_conv_diff.copy()
        vRHS_conv_diff_p = vRHS_conv_diff.copy()

    for stepcount in range(3, nt + 1):

        ustar, vstar, uRHS_conv_diff, vRHS_conv_diff = projection_method.step1(u, v, nx, ny, nu, x, y, xx, yy, dx, dy,
                                                                               dt, epsilon, F, R, theta, r,
                                                                               uRHS_conv_diff_p, uRHS_conv_diff_pp,
                                                                               vRHS_conv_diff_p, vRHS_conv_diff_pp,
                                                                               dpdx, dpdy,bc)

        # Step2
        ustarstar, vstarstar = projection_method.step2(ustar, vstar, dpdx, dpdy, dt)

        # Step3
        p = projection_method.step3(ustarstar, vstarstar, rho, epsilon, dx, dy, nx, ny, nx_sp, ny_sp, K, dt)

        # Step4
        u, v, dpdx, dpdy = projection_method.step4(ustarstar, vstarstar, p, dx, dy, nx, ny, dt)

        if np.mod(stepcount, saveth_iter) == 0:
            ip_op.write_szl_2D(xx, yy, p, u, v, stepcount * dt, int(stepcount / saveth_iter))

        print(stepcount)

        uRHS_conv_diff_pp = uRHS_conv_diff_p.copy()
        vRHS_conv_diff_pp = vRHS_conv_diff_p.copy()

        uRHS_conv_diff_p = uRHS_conv_diff.copy()
        vRHS_conv_diff_p = vRHS_conv_diff.copy()

        '''
        fig = pyplot.figure(figsize=(11, 7), dpi=100)
        pyplot.quiver(xx[::3, ::3], yy[::3, ::3], u[::3, ::3], v[::3, ::3])
        
        fig = pyplot.figure(figsize=(11, 7), dpi=100)
        pyplot.quiver(xx, yy, u, v)
        
        pyplot.show()
        '''


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
