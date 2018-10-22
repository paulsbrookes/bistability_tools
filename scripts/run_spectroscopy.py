import os, sys
from cqed_tools.calibration import *
import matplotlib.pyplot as plt


display = False


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
    return t_array

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
    return c_array


def peak_gen(eps_array):
    n_spectra = eps_array.shape[0]
    peak_array = np.zeros(n_spectra, dtype=np.float64)
    peak_array += 10.4761 * (eps_array <= 1e-5)
    peak_array += 10.4761 * (1e-5 < eps_array) * (eps_array <= 1e-4)
    peak_array += 10.476 * (1e-4 < eps_array) * (eps_array <= 1e-3)
    peak_array += 10.475 * (1e-3 < eps_array) * (eps_array <= 2e-3)
    peak_array += 10.474 * (2e-3 < eps_array) * (eps_array <= 3e-3)
    peak_array += 10.4732 * (3e-3 < eps_array) * (eps_array <= 4e-3)
    peak_array += 10.4726 * (4e-3 < eps_array) * (eps_array <= 5e-3)
    peak_array += 10.4721 * (5e-3 < eps_array) * (eps_array <= 6e-3)
    peak_array += 10.4716 * (6e-3 < eps_array) * (eps_array <= 7e-3)
    peak_array += 10.4712 * (7e-3 < eps_array) * (eps_array <= 8e-3)
    return peak_array


def dip_gen(eps_array):
    n_spectra = eps_array.shape[0]
    dip_array = np.zeros(n_spectra, dtype=np.float64)
    dip_array += 10.475 * (eps_array <= 1e-5)
    dip_array += 10.475 * (1e-5 < eps_array) * (eps_array <= 1e-4)
    dip_array += 10.475 * (1e-4 < eps_array) * (eps_array <= 1e-3)
    dip_array += 10.474 * (1e-3 < eps_array) * (eps_array <= 2e-3)
    dip_array += 10.4733 * (2e-3 < eps_array) * (eps_array <= 3e-3)
    dip_array += 10.4725 * (3e-3 < eps_array) * (eps_array <= 4e-3)
    dip_array += 10.4719 * (4e-3 < eps_array) * (eps_array <= 5e-3)
    dip_array += 10.4712 * (5e-3 < eps_array) * (eps_array <= 6e-3)
    dip_array += 10.4705 * (6e-3 < eps_array) * (eps_array <= 7e-3)
    dip_array += 10.4698 * (7e-3 < eps_array) * (eps_array <= 8e-3)
    return dip_array




if __name__ == '__main__':
    fd_lower = 10.468
    fd_upper = 10.472
    threshold = 0.15
    fd = None

    #eps_array = np.array([0.01,0.1,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])*1e-3

    #eps_array = np.array([7.98270444e-05, 8.95674170e-05, 1.00496295e-04, 1.12758697e-04,
    #   1.26517339e-04, 1.41954789e-04, 1.59275893e-04, 1.78710492e-04,
    #   2.00516470e-04, 2.24983179e-04, 2.52435279e-04, 2.83237042e-04,
    #   3.17797188e-04, 3.56574309e-04, 4.00082956e-04, 4.48900459e-04,
    #   5.03674600e-04, 5.65132196e-04, 6.34088753e-04, 7.11459282e-04,
    #   7.98270444e-04, 8.95674170e-04, 1.00496295e-03, 1.12758697e-03,
    #   1.26517339e-03, 1.41954789e-03, 1.59275893e-03, 1.78710492e-03,
    #   2.00516470e-03, 2.24983179e-03, 2.52435279e-03, 2.83237042e-03,
    #   3.17797188e-03, 3.56574309e-03, 4.00082956e-03, 4.48900459e-03,
    #   5.03674600e-03, 5.65132196e-03, 6.34088753e-03, 7.11459282e-03,
    #   7.98270444e-03])

    eps_array = np.array([0.00266514, 0.00304213, 0.00333483, 0.00373371, 0.00443115,
       0.00494409, 0.00546806, 0.00607245, 0.00678645, 0.00761325])


    t_array = t_gen(eps_array)
    c_array = c_gen(eps_array)
    peak_array = peak_gen(eps_array)
    dip_array = dip_gen(eps_array)
    difference_array = peak_array - dip_array
    fd_upper_array = peak_array + difference_array
    fd_lower_array = dip_array - 2.5*difference_array

    for idx, eps in enumerate(eps_array):

        t_levels = t_array[idx]
        c_levels = c_array[idx]
        fd_lower = fd_lower_array[idx]
        fd_upper = fd_upper_array[idx]

        params = DefaultParameters()
        params.eps = eps
        params.t_levels = t_levels
        params.c_levels = c_levels
        multi_results = multi_sweep(np.array([eps]), fd_lower, fd_upper, params, threshold, custom=False)

        labels = params.labels

        collected_data_re = None
        collected_data_im = None
        collected_data_abs = None
        collected_data_edge_occupation_t = None
        results_list = []
        for subsweep in multi_results.values():
            p = subsweep['params'].iloc[0]
            truncated_results = subsweep.drop(columns=['states'])
            directory = 't' + str(p.t_levels) + 'c' + str(p.c_levels) + 'eps=' + str(eps) + 'GHz'
            if not os.path.exists(directory):
                os.makedirs(directory)
            truncated_results.to_hdf(directory+'/results.h5','results')
            subsweep[['fd_points','states']].to_hdf(directory+'/states.h5','states')
            qsave(p, directory+'/params')


    #if display:
    #    fig, axes = plt.subplots(1, 1)
    #    collected_dataset['a_abs'].squeeze().to_dataframe()['a_abs'].plot(ax=axes)
    #    plt.show()
