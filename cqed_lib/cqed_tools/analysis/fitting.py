import xarray as xr
from scipy.optimize import curve_fit
from .loading import *
from ..tools import *


def decay(t, ar, ai, br, bi, T):
    signal_r = ar - br * np.exp(-t / T)
    signal_i = ai - bi * np.exp(-t / T)
    signal = np.array([signal_r, signal_i])
    return signal


def decay_flat(t, ar, ai, br, bi, T):
    index = int(len(t) / 2)
    return decay(t[0:index], ar, ai, br, bi, T).flatten()


def decay_gen(a_ss):
    def decay_fixed(t, A, T):
        alpha = a_ss + A * np.exp(-t / T)
        return alpha

    return decay_fixed


def analyse_directory(directory, results_list, flags):
    time_constants, collated_popt, collated_a_ss, collated_states = results_list
    dims = ['eps', 'fd', 'qubit_state', 't_levels', 'c_levels', 'n_c']
    settings, state, results = load_results(directory)

    if state is not None:
        t_levels = int(float(settings.t_levels.values[0]))
        c_levels = int(float(settings.c_levels.values[0]))
        a = tensor(destroy(c_levels), qeye(t_levels))
        a_ss = expect(state, a)

        coords = [[float(settings.eps)], [float(settings.fd)], [int(float(settings.qubit_state))], [t_levels],
                  [c_levels], [float(settings.n_c)]]

        a_ss_point = xr.DataArray([[[[[[a_ss]]]]]], coords=coords, dims=dims)
        state_point = xr.DataArray([[[[[[state]]]]]], coords=coords, dims=dims)

        if collated_a_ss is None:
            collated_a_ss = a_ss_point
            collated_states = state_point
        else:
            collated_a_ss = collated_a_ss.combine_first(a_ss_point)
            collated_states = collated_states.combine_first(state_point)

    if results is not None and 'spectrum' not in flags:
        i = results.a_op_re

        t_end = i.index[-1]
        if t_end > 4.0:

            t0 = 3.0
            i_truncated = i[t0:]
            times = i_truncated.index

            d_est = (i_truncated.iloc[-1] - i_truncated.iloc[0]) / (times[-1] - times[0])
            T_est = (a_ss.real - i_truncated.iloc[0]) / d_est
            A_est = -d_est * T_est * np.exp(times[0] / T_est)

            decay_fixed = decay_gen(a_ss.real)

            popt, pcov = curve_fit(f=decay_fixed, xdata=times, ydata=i_truncated, p0=[A_est, T_est])

            T_constant = popt[1]

            point = xr.DataArray([[[[[[T_constant]]]]]], coords=coords, dims=dims)

            popt_point = xr.DataArray([[[[[[popt]]]]]], coords=(coords + [np.arange(2)]), dims=dims + ['popt'])

            if time_constants is None:
                time_constants = point
                collated_popt = popt_point
            else:
                time_constants = time_constants.combine_first(point)
                collated_popt = collated_popt.combine_first(popt_point)

    return [time_constants, collated_popt, collated_a_ss, collated_states]


def analyse_tree(directory, use_flags=True, save=True, load=True):

    cache_dir = 'cache'

    if load:
        if os.path.exists(cache_dir):
            results = xr.open_dataset(cache_dir+'/simulated_results.nc')
            time_constants = results['constants']

            collated_a_ss_dataset = xr.open_dataset(cache_dir+'/a_ss.nc')
            collated_a_ss = collated_a_ss_dataset['a_r'] + 1j * collated_a_ss_dataset['a_i']

            collated_popt = xr.open_dataset(cache_dir+'/collated_popt.nc')

            collated_directories = qload(cache_dir+'/collated_directories')

            return time_constants, collated_a_ss, collated_popt, collated_directories

    time_constants = None
    collated_popt = None
    collated_a_ss = None
    collated_states = None
    collated_directories = None

    results_list = [time_constants, collated_popt, collated_a_ss, collated_states]

    walk = os.walk(directory)

    for content in walk:

        directory = content[0]

        print(directory)

        if os.path.exists(directory + '/flags.txt') and use_flags:
            print('flags!')
            with open(directory + '/flags.txt', 'r') as f:
                line = f.readline().split('\n')[0]
                flags = line.split(',')
        else:
            flags = []

        if 'ignore' in flags:
            content[1][:] = []

        elif 'spectrum' in flags:
            content[1][:] = []
            sub_walk = os.walk(directory)

            for sub_content in sub_walk:
                sub_directory = sub_content[0]

                if os.path.exists(sub_directory + '/flags.txt') and use_flags:
                    with open(sub_directory + '/flags.txt', 'r') as f:
                        line = f.readline().split('\n')[0]
                        sub_flags = line.split(',') + flags
                else:
                    sub_flags = [] + flags

                if 'ignore' in sub_flags:
                    sub_content[1][:] = []
                else:
                    time_constants, collated_popt, collated_a_ss, collated_states = analyse_directory(sub_directory,
                                                                                                      results_list,
                                                                                                      sub_flags)
                    results_list = [time_constants, collated_popt, collated_a_ss, collated_states]

        else:
            time_constants, collated_popt, collated_a_ss, collated_states = analyse_directory(directory, results_list,
                                                                                              flags)
            results_list = [time_constants, collated_popt, collated_a_ss, collated_states]

        if os.path.exists(directory + '/settings.csv'):
            settings = pd.read_csv(directory + '/settings.csv')
            settings = settings.set_index('job_index').T
            tuples = [(float(settings.eps.values[0]), float(settings.fd.values[0]))]
            index = pd.MultiIndex.from_tuples(tuples, names=['eps', 'fd'])
            directory_idx = os.path.basename(os.path.normpath(directory))
            packaged_index = pd.Series(directory_idx, index=index)
            if collated_directories is not None:
                collated_directories = collated_directories.combine_first(packaged_index)
            else:
                collated_directories = packaged_index

    if save:

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        else:
            os.remove(cache_dir+'/simulated_results.nc')
            os.remove(cache_dir+'/a_ss.nc')
            os.remove(cache_dir+'/collated_popt.nc')
            os.remove(cache_dir+'/collated_directories.qu')

        results_dict = dict()
        results_dict['constants'] = time_constants
        results = xr.Dataset(results_dict)
        results.to_netcdf(cache_dir+'/simulated_results.nc')

        collated_a_ss_dict = {'a_r': collated_a_ss.real, 'a_i': collated_a_ss.imag}
        collated_a_ss_dataset = xr.Dataset(collated_a_ss_dict)
        collated_a_ss_dataset.to_netcdf(cache_dir+'/a_ss.nc')

        collated_popt.to_netcdf(cache_dir+'/collated_popt.nc')

        qsave(collated_directories, cache_dir+'/collated_directories')

    return time_constants, collated_a_ss, collated_popt, collated_directories


def analyse_tree_liouvillian(directory, use_flags=True, save=True, load=True):

    if load and os.path.exists('results.h5'):
        print('Loading.')
        results = pd.read_hdf('results.h5')
        return results

    results = None

    walk = os.walk(directory)

    for content in walk:

        directory = content[0]

        print(directory)

        if os.path.exists(directory + '/flags.txt') and use_flags:
            print('flags!')
            with open(directory + '/flags.txt', 'r') as f:
                line = f.readline().split('\n')[0]
                flags = line.split(',')
        else:
            flags = []

        try:
            if os.path.exists(directory + '/steady_state.qu'):
                steady_state = qload(directory + '/steady_state')
                settings = load_settings(directory + '/settings.csv')
                t_levels = int(float(settings.t_levels))
                c_levels = int(float(settings.c_levels))
                a = tensor(destroy(c_levels), qeye(t_levels))
                a_ss = expect(steady_state, a)

                if os.path.exists(directory + '/steady_state.qu') and os.path.exists(
                        directory + '/state_checkpoint.qu'):
                    time_constant = calculate_constants(directory)
                else:
                    time_constant = None

                mi = pd.MultiIndex.from_arrays(np.array([settings.values]).T, names=tuple(settings.index))
                row = pd.DataFrame(np.array([[time_constant, a_ss]]), columns=['time_constants', 'a_ss'], index=mi)
                if results is not None:
                    results = pd.concat([results, row])
                else:
                    results = row
        except:
            print('Failure.')

    dtypes = dict()
    dtypes['time_constants'] = np.complex
    dtypes['a_ss'] = np.complex
    results = results.astype(dtypes)

    if save:
        results.to_hdf('results_liouvillian.h5', key='results')

    return results