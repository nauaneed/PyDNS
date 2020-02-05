from tecio.tecio_szl import create_ordered_zone, open_file, zone_write_double_values, close_file, zone_set_solution_time, FD_DOUBLE
import os

def write_szl_2D(X, Y, p, u, v, t, snap_num):
    var_names = ['X', 'Y', 'p', 'u', 'v']
    outFilename = "snapshot%04d.szplt" % (snap_num)
    os.chdir('data/')
    file_handle = open_file(outFilename, "PyDNS_data", var_names)
    zone = create_ordered_zone(file_handle, "Zone", (X.shape[1], X.shape[0], 1), None,
                               [FD_DOUBLE for i in range(len(var_names))])
    zone_set_solution_time(file_handle, zone, t, 1)
    zone_write_double_values(file_handle, zone, 1, X.flatten())
    zone_write_double_values(file_handle, zone, 2, Y.flatten())
    zone_write_double_values(file_handle, zone, 3, p.flatten())
    zone_write_double_values(file_handle, zone, 4, u.flatten())
    zone_write_double_values(file_handle, zone, 5, v.flatten())
    close_file(file_handle)
    os.chdir('../')