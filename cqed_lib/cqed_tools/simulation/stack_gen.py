from scipy import interpolate
import numpy as np

power_list = [-18, -17, -16, -15, -14, -13, -12]
fd1_list = [10.4724, 10.4720, 10.4710, 10.4705, 10.4700, 10.469, 10.468]
fd2_list = [10.4734, 10.4732, 10.4728, 10.4725, 10.4723, 10.4723, 10.4720]
fd_peak_list = []

fd1_func = interpolate.interp1d(power_list, fd1_list)
fd2_func = interpolate.interp1d(power_list, fd2_list)


def frequencies_gen(fd0, fd1, fd2, fd3, df0, df1, df2):
    frequencies1 = np.arange(fd0, fd1 + df0, df0)
    frequencies2 = np.arange(fd1, fd2 + df1, df1)
    frequencies3 = np.arange(fd3, fd2 - df2, -df2)
    frequencies = np.hstack([frequencies1, frequencies2, frequencies3])
    frequencies = np.round(frequencies, 10)
    frequencies = np.array(sorted(set(list(frequencies))))
    return frequencies


def t_gen(eps_array):
    n_spectra = eps_array.shape[0]
    t_array = np.zeros(n_spectra, dtype=np.int)
    t_array += 3 * (eps_array <= 1e-5)
    t_array += 4 * (1e-5 < eps_array) * (eps_array <= 1e-4)
    t_array += 6 * (1e-4 < eps_array) * (eps_array <= 1e-3)
    t_array += 7 * (1e-3 < eps_array) * (eps_array <= 2e-3)
    t_array += 8 * (2e-3 < eps_array) * (eps_array <= 3e-3)
    t_array += 9 * (3e-3 < eps_array) * (eps_array <= 4e-3)
    t_array += 9 * (4e-3 < eps_array) * (eps_array <= 5e-3)
    t_array += 9 * (5e-3 < eps_array) * (eps_array <= 6e-3)
    t_array += 9 * (6e-3 < eps_array) * (eps_array <= 7e-3)
    t_array += 9 * (7e-3 < eps_array) * (eps_array <= 8e-3)
    t_array += 10 * (8e-3 < eps_array)
    return t_array - 2


def c_gen(eps_array):
    n_spectra = eps_array.shape[0]
    c_array = np.zeros(n_spectra, dtype=np.int)
    c_array += 3 * (eps_array <= 1e-5)
    c_array += 5 * (1e-5 < eps_array) * (eps_array <= 1e-4)
    c_array += 11 * (1e-4 < eps_array) * (eps_array <= 1e-3)
    c_array += 20 * (1e-3 < eps_array) * (eps_array <= 2e-3)
    c_array += 30 * (2e-3 < eps_array) * (eps_array <= 3.0e-3)
    c_array += 40 * (3e-3 < eps_array) * (eps_array <= 4e-3)
    c_array += 50 * (4e-3 < eps_array) * (eps_array <= 5e-3)
    c_array += 55 * (5e-3 < eps_array) * (eps_array <= 6e-3)
    c_array += 65 * (6e-3 < eps_array) * (eps_array <= 7e-3)
    c_array += 75 * (7e-3 < eps_array) * (eps_array <= 8e-3)
    c_array += 80 * (8e-3 < eps_array)
    return c_array - 5
