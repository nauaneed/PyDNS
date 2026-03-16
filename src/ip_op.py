from pathlib import Path

import numpy as np


DATA_DIR = Path("data")


def _get_output_path(filename):
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / filename


def write_snapshot_2D(X, Y, p, u, v, t, snap_num, formats=("szl",)):
    for output_format in formats:
        normalized_format = output_format.lower()
        if normalized_format == "szl":
            write_szl_2D(X, Y, p, u, v, t, snap_num)
        elif normalized_format == "vtk":
            write_vtk_2D(X, Y, p, u, v, t, snap_num)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def write_szl_2D(X, Y, p, u, v, t, snap_num):
    from tecio.tecio_szl import create_ordered_zone, open_file, zone_write_double_values, close_file, zone_set_solution_time, FD_DOUBLE

    var_names = ['X', 'Y', 'p', 'u', 'v']
    outFilename = "snapshot%04d.szplt" % (snap_num)
    output_path = _get_output_path(outFilename)
    file_handle = open_file(str(output_path), "PyDNS_data", var_names)
    zone = create_ordered_zone(file_handle, "Zone", (X.shape[1], X.shape[0], 1), None,
                               [FD_DOUBLE for i in range(len(var_names))])
    zone_set_solution_time(file_handle, zone, t, 1)
    zone_write_double_values(file_handle, zone, 1, X.flatten())
    zone_write_double_values(file_handle, zone, 2, Y.flatten())
    zone_write_double_values(file_handle, zone, 3, p.flatten())
    zone_write_double_values(file_handle, zone, 4, u.flatten())
    zone_write_double_values(file_handle, zone, 5, v.flatten())
    close_file(file_handle)


def write_vtk_2D(X, Y, p, u, v, t, snap_num):
    output_path = _get_output_path("snapshot%04d.vtk" % (snap_num))
    ny, nx = X.shape
    points = np.column_stack((X.ravel(), Y.ravel(), np.zeros(nx * ny)))
    velocity = np.column_stack((u.ravel(), v.ravel(), np.zeros(nx * ny)))

    with output_path.open("w", encoding="ascii") as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write(f"PyDNS snapshot at t={t:.6f}\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET STRUCTURED_GRID\n")
        vtk_file.write(f"DIMENSIONS {nx} {ny} 1\n")
        vtk_file.write(f"POINTS {nx * ny} float\n")
        for x_coord, y_coord, z_coord in points:
            vtk_file.write(f"{x_coord:.16e} {y_coord:.16e} {z_coord:.16e}\n")

        vtk_file.write("FIELD FieldData 1\n")
        vtk_file.write("TIME 1 1 double\n")
        vtk_file.write(f"{t:.16e}\n")

        vtk_file.write(f"POINT_DATA {nx * ny}\n")
        vtk_file.write("SCALARS pressure float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for value in p.ravel():
            vtk_file.write(f"{value:.16e}\n")

        vtk_file.write("VECTORS velocity float\n")
        for u_value, v_value, w_value in velocity:
            vtk_file.write(f"{u_value:.16e} {v_value:.16e} {w_value:.16e}\n")