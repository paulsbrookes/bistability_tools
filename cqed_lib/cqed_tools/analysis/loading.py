import pandas as pd
import numpy as np
from qutip import *
import os

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


def duplicate_pruner(series, level):
    multi_index = series.index
    names = multi_index.names
    index = multi_index.names.index(level)
    levels_to_drop = names[index + 1:]

    dropped = multi_index.droplevel(levels_to_drop)
    pruned_multi_indices = dropped.drop_duplicates()

    pruned_series = None
    n_entries = pruned_multi_indices.shape[0]

    for i in range(n_entries):
        pruned_multi_index = pruned_multi_indices[i]
        sub_series = series[pruned_multi_index]
        sub_multi_indices = sub_series.index
        last_sub_multi_index = sub_multi_indices[-1]
        pruned_sub_series = sub_series[last_sub_multi_index]
        combined_tuple = tuple(pruned_multi_index) + last_sub_multi_index
        combined_multi_index = pd.MultiIndex.from_tuples([combined_tuple], names=multi_index.names)
        pruned_sub_series = pd.Series(pruned_sub_series, index=combined_multi_index)
        if pruned_series is None:
            pruned_series = pruned_sub_series
        else:
            pruned_series = pruned_series.append(pruned_sub_series)

    return pruned_series


def load_results(directory):
    settings = None
    state = None
    results = None

    if os.path.exists(directory + '/steady_state.qu'):

        settings = pd.read_csv(directory + '/settings.csv')
        settings = settings.set_index('job_index').T
        state = qload(directory + '/steady_state')

        if os.path.exists(directory + '/results.csv'):
            results = pd.read_csv(directory + '/results.csv')
            results.times /= (2 * np.pi * 1000)
            results = results.set_index('times')

    return settings, state, results


def load_settings(settings_path):
    settings = pd.read_csv(settings_path, header=None)

    settings = settings.set_index(0)
    settings = settings.T

    dtypes = dict()
    dtypes['job_index'] = np.int
    dtypes['eps'] = np.float
    dtypes['fd'] = np.float
    dtypes['qubit_state'] = np.int
    dtypes['t_levels'] = np.int
    dtypes['c_levels'] = np.int
    dtypes['fc'] = np.float
    dtypes['Ej'] = np.float
    dtypes['g'] = np.float
    dtypes['Ec'] = np.float
    dtypes['kappa'] = np.float
    dtypes['gamma'] = np.float
    dtypes['gamma_phi'] = np.float
    dtypes['n_t'] = np.float
    dtypes['n_c'] = np.float
    dtypes['end_time'] = np.float
    dtypes['snapshots'] = np.int
    dtypes['group_folder'] = str

    settings = settings.astype(dtypes)
    settings = settings.iloc[0, :]

    return settings