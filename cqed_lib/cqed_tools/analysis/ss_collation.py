from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def ss_results_collator(path):
    
    collated_results_dict = OrderedDict()
    collated_popt = None
    
    dtypes = {'eps': np.float64,
             'fd': np.float64,
             'qubit_state': np.int32,
             't_levels': np.int32,
             'c_levels': np.int32,
             'fc': np.float64,
             'Ec': np.float64,
             'Ej': np.float64,
             'g': np.float64,
             'kappa': np.float64,
             'gamma': np.float64,
             'gamma_phi': np.float64,
             'n_t': np.float64,
             'n_c': np.float64,
             'end_time': np.float64,
             'snapshots': np.int32,
             'group_folder': str}

    walk = os.walk(path)
    for content in walk:
        directory = content[0]
        if os.path.exists(directory+'/ss_results.csv'):

            settings = pd.read_csv(directory+'/settings.csv')
            settings = settings.set_index('job_index')
            settings = settings.T
            settings = settings.astype(dtypes)

            results = pd.read_csv(directory+'/ss_results.csv')
            results = results.set_index('job_index')
            measurement_names = results.columns

            n_coords = 14
            coords = settings.iloc[0,0:n_coords]
            packaged_coords = []
            for coord in coords:
                packaged_coords.append([coord])

            for i, measurement in enumerate(measurement_names[0:2]):

                value = results.iloc[0,i]
                packaged_value = value
                for i in range(n_coords):
                    packaged_value = [packaged_value]
                value_xarray = xr.DataArray(packaged_value, coords=packaged_coords, dims=coords.index)

                if measurement in collated_results_dict:
                    collated_results_dict[measurement] = collated_results_dict[measurement].combine_first(value_xarray)
                else:
                    collated_results_dict[measurement] = value_xarray

    collated_results = xr.Dataset(collated_results_dict)
    
    return collated_results


def hilbert_calibration_plotter(collated_results):

    import itertools
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 

    fig, axes = plt.subplots(1,1)

    legend = []

    a_op_re = collated_results['a_op_re'].squeeze()
    a_op_im = collated_results['a_op_im'].squeeze()
    a_op_abs = np.abs(a_op_re + 1j*a_op_im)
    n_t_levels = a_op_re.shape[1]
    n_c_levels = a_op_re.shape[2]
    for i in range(n_t_levels):
        for j in range(n_c_levels):
            sweep = a_op_abs[:,i,j].dropna('fd')
            if sweep.shape[0] != 0:
                legend.append(str(sweep.t_levels.values[()]) + 'x' + str(sweep.c_levels.values[()]))
                sweep.plot.line(marker=marker.next(), ax=axes)

    axes.legend(legend)

    plt.show()


