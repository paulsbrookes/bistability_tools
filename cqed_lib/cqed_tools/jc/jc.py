from qutip import *
from ..mf import *
import pandas as pd
from scipy.interpolate import interp1d
from copy import deepcopy
import matplotlib.pyplot as plt


def ham_gen_jc(params, alpha=0):
    sz = tensor(sigmaz(), qeye(params.c_levels))
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels)) + alpha
    ham = (params.fc-params.fd)*a.dag()*a
    ham += params.eps*(a+a.dag())
    ham += 0.5*(params.f01-params.fd)*sz
    ham += params.g*(a*sm.dag() + a.dag()*sm)
    ham *= 2*np.pi
    return ham


def c_ops_gen_jc(params, alpha=0):
    c_ops = []
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels)) + alpha
    if params.gamma > 0.0:
        c_ops.append(np.sqrt(2*np.pi*params.gamma)*sm)
        if params.n_t > 0:
            c_ops.append(np.sqrt(2*np.pi*params.gamma*(1+params.n_t))*sm.dag())
    if params.gamma_phi > 0.0:
        c_ops.append(np.sqrt(2*np.pi*params.gamma_phi)*sm.dag()*sm)
    if params.kappa > 0.0:
        c_ops.append(np.sqrt(2*np.pi*params.kappa)*a)
        if params.n_c > 0:
            c_ops.append(np.sqrt(2*np.pi*params.gamma*(1+params.n_c))*a.dag())
    return c_ops


def iterative_alpha_calc(params, n_cycles=10, initial_alpha=0):
    alpha = initial_alpha

    try:

        for idx in range(n_cycles):
            ham = ham_gen_jc(params, alpha=alpha)
            c_ops = c_ops_gen_jc(params, alpha=alpha)
            rho = steadystate(ham, c_ops)

            a = tensor(qeye(2), destroy(params.c_levels)) + alpha

            a_exp = expect(a, rho)

            alpha = a_exp

    except:
        alpha = None

    return alpha


class Spectrum:
    def __init__(self, parameters):
        print('hello')
        self.parameters = deepcopy(parameters)
        self.mf_amplitude = None
        self.me_amplitude = None
        self.transmission_exp = None

    def iterative_calculate(self, fd_array, initial_alpha=0, n_cycles=10, prune=True):

        if self.parameters.fc < self.parameters.f01:
            change = 'hard'
        else:
            change = 'soft'

        params = deepcopy(self.parameters)

        fd_array = np.sort(fd_array)
        a_array = np.zeros(fd_array.shape[0], dtype=complex)
        alpha = initial_alpha
        for fd_idx, fd in tqdm(enumerate(fd_array)):
            params.fd = fd
            alpha = iterative_alpha_calc(params, initial_alpha=alpha, n_cycles=n_cycles)
            a_array[fd_idx] = alpha

        if change is 'hard':
            alpha_bright_iterative = pd.Series(a_array, index=fd_array, name='alpha_bright')
        else:
            alpha_dim_iterative = pd.Series(a_array, index=fd_array, name='alpha_dim')

        fd_array = np.flip(fd_array)
        a_array = np.zeros(fd_array.shape[0], dtype=complex)
        alpha = initial_alpha
        for fd_idx, fd in tqdm(enumerate(fd_array)):
            params.fd = fd
            alpha = iterative_alpha_calc(params, initial_alpha=alpha, n_cycles=n_cycles)
            a_array[fd_idx] = alpha

        if change is 'hard':
            alpha_dim_iterative = pd.Series(a_array, index=fd_array, name='alpha_dim')
        else:
            alpha_bright_iterative = pd.Series(a_array, index=fd_array)

        if prune:
            alpha_dim_iterative = alpha_dim_iterative.dropna()
            alpha_dim_iterative.sort_index(inplace=True)
            # alpha_dim_diff = np.diff(alpha_dim_iterative)/np.diff(alpha_dim_iterative.index)
            # first_dim_idx = np.argmax(np.abs(alpha_dim_diff)) + 1
            first_dim_idx = np.argmax(alpha_dim_iterative.real)
            alpha_dim_iterative = alpha_dim_iterative.iloc[first_dim_idx:]

            alpha_bright_iterative = alpha_bright_iterative.dropna()
            alpha_bright_iterative.sort_index(inplace=True)
            # alpha_bright_diff = np.diff(alpha_bright_iterative) / np.diff(alpha_bright_iterative.index)
            # last_bright_idx = np.argmax(np.abs(alpha_bright_diff))
            last_bright_idx = np.argmin(alpha_bright_iterative.imag)
            alpha_bright_iterative = alpha_bright_iterative.iloc[:last_bright_idx + 1]

        self.iterative_amplitude = pd.concat([alpha_dim_iterative, alpha_bright_iterative], axis=1)

    def gen_iterative_hilbert_params(self, fd_limits, kind='linear', fill_value='extrapolate', fraction=0.5,
                                     level_scaling=1.0, max_shift=False, max_levels=True):

        alpha_dim = self.iterative_amplitude['alpha_dim'].dropna()
        # alpha_dim.sort_index(inplace=True)
        # alpha_dim_diff = np.diff(alpha_dim)/np.diff(alpha_dim.index)
        # first_dim_idx = np.argmax(np.abs(alpha_dim_diff)) + 1
        # alpha_dim = alpha_dim.iloc[first_dim_idx:]

        alpha_bright = self.iterative_amplitude['alpha_bright'].dropna()
        # alpha_bright.sort_index(inplace=True)
        # alpha_bright_diff = np.diff(alpha_bright) / np.diff(alpha_bright.index)
        # last_bright_idx = np.argmax(np.abs(alpha_bright_diff))
        # alpha_bright = alpha_bright.iloc[:last_bright_idx]

        new_iterative_alphas = pd.concat([alpha_dim, alpha_bright], axis=1)
        self.iterative_amplitude = new_iterative_alphas

        alpha_dim_real_func = interpolate.interp1d(alpha_dim.index, alpha_dim.real, kind=kind, fill_value=fill_value)
        alpha_dim_imag_func = interpolate.interp1d(alpha_dim.index, alpha_dim.imag, kind=kind, fill_value=fill_value)

        def alpha_dim_func_single(fd):
            alpha_dim = alpha_dim_real_func(fd) + 1j * alpha_dim_imag_func(fd)
            return alpha_dim

        alpha_dim_func_vec = np.vectorize(alpha_dim_func_single)

        def alpha_dim_func(fd_array):
            alpha_dim_array = alpha_dim_func_vec(fd_array)
            alpha_dim_series = pd.Series(alpha_dim_array, index=fd_array, name='alpha_dim_func')
            return alpha_dim_series

        alpha_bright_real_func = interpolate.interp1d(alpha_bright.index, alpha_bright.real, kind=kind,
                                                      fill_value=fill_value)
        alpha_bright_imag_func = interpolate.interp1d(alpha_bright.index, alpha_bright.imag, kind=kind,
                                                      fill_value=fill_value)

        def alpha_bright_func_single(fd):
            alpha_bright = alpha_bright_real_func(fd) + 1j * alpha_bright_imag_func(fd)
            return alpha_bright

        alpha_bright_func_vec = np.vectorize(alpha_bright_func_single)

        def alpha_bright_func(fd_array):
            alpha_bright_array = alpha_bright_func_vec(fd_array)
            alpha_bright_series = pd.Series(alpha_bright_array, index=fd_array, name='alpha_bright')
            return alpha_bright_series

        alpha_dim_interp = alpha_dim_func(fd_array)
        alpha_bright_interp = alpha_bright_func(fd_array)
        alpha_diff_interp = (alpha_bright_interp - alpha_dim_interp).dropna()
        if max_shift:
            min_diff = np.min(np.abs(alpha_diff_interp))
            alpha_diff_unit_interp = alpha_diff_interp / np.abs(alpha_diff_interp)
            alpha_0_interp = alpha_dim_interp + fraction * min_diff * alpha_diff_unit_interp
        else:
            alpha_0_interp = alpha_dim_interp + fraction * alpha_diff_interp
        alpha_diff_interp.name = 'alpha_diff'
        alpha_0_interp.name = 'alpha_0'
        hilbert_params = pd.concat([alpha_diff_interp, alpha_0_interp], axis=1)

        if max_levels:
            min_diff = np.min(np.abs(alpha_diff_interp))
            hilbert_params['c_levels'] = np.int(np.ceil(level_scaling * min_diff ** 2))
        else:
            hilbert_params['c_levels'] = np.ceil(level_scaling * np.abs(alpha_diff_interp.values) ** 2).astype(int)

        hilbert_params['c_levels'].loc[:fd_limits[0]] = self.parameters.c_levels
        hilbert_params['alpha_0'].loc[:fd_limits[0]] = self.iterative_amplitude['alpha_bright'].loc[:fd_limits[0]]
        hilbert_params['c_levels'].loc[fd_limits[1]:] = self.parameters.c_levels
        hilbert_params['alpha_0'].loc[fd_limits[1]:] = self.iterative_amplitude['alpha_dim'].loc[fd_limits[1]:]

        # hilbert_params = pd.concat([hilbert_params, alpha_dim_interp, alpha_bright_interp], axis=1)

        self.hilbert_params = hilbert_params

    def mf_calculate(self, fd_array, characterise_only=False):
        if self.mf_amplitude is None:
            self.mf_amplitude = map_mf_jc(self.parameters, fd_array=fd_array, characterise_only=characterise_only)
        else:
            fd0 = fd_array[0]
            fd1 = fd_array[-1]
            idx0 = self.mf_amplitude.index.get_loc(fd0, method='nearest')
            idx1 = self.mf_amplitude.index.get_loc(fd1, method='nearest')
            alpha0_dim = self.mf_amplitude['a_dim'].iloc[idx0]
            sm0_dim = self.mf_amplitude['sm_dim'].iloc[idx0]
            sz0_dim = self.mf_amplitude['sz_dim'].iloc[idx0]
            alpha0_bright = self.mf_amplitude['a_bright'].iloc[idx1]
            sm0_bright = self.mf_amplitude['sm_bright'].iloc[idx1]
            sz0_bright = self.mf_amplitude['sz_bright'].iloc[idx1]
            mf_amplitude_new = mf_characterise_jc(self.parameters, fd_array, alpha0_bright=alpha0_bright,
                                                  sm0_bright=sm0_bright, sz0_bright=sz0_bright, alpha0_dim=alpha0_dim,
                                                  sm0_dim=sm0_dim, sz0_dim=sz0_dim, check_bistability=False)
            self.mf_amplitude = pd.concat([self.mf_amplitude, mf_amplitude_new])

        self.mf_amplitude = self.mf_amplitude.sort_index()
        self.mf_amplitude = self.mf_amplitude[~self.mf_amplitude.index.duplicated(keep='first')]

    def generate_hilbert_params(self, c_levels_bi_scale=1.0, scale=0.5, fd_limits=None, max_shift=True,
                                c_levels_mono=10, c_levels_bi=10, alpha_0_mono=0, alpha_0_bi=0, kind='linear',
                                method='extrapolate_alpha_0'):
        print(c_levels_bi)
        self.hilbert_params = generate_hilbert_params(self.mf_amplitude, c_levels_bi_scale=c_levels_bi_scale,
                                                      scale=scale, fd_limits=fd_limits, kind=kind,
                                                      max_shift=max_shift, c_levels_mono=c_levels_mono,
                                                      c_levels_bi=c_levels_bi, alpha_0_mono=alpha_0_mono,
                                                      alpha_0_bi=alpha_0_bi, method=method)

    def me_calculate(self, solver_kwargs={}, c_levels_bi_scale=1.0, scale=0.5, fd_limits=None, fill_value='extrapolate',
                     max_shift=False, c_levels_mono=10, c_levels_bi=10, alpha_0_mono=0, alpha_0_bi=0, kind='linear',
                     method='extrapolate_alpha_0', level_scaling=1.0, max_levels=True):

        if method is 'iterative':
            frequencies = self.iterative_amplitude.index
            self.gen_iterative_hilbert_params(fd_limits, kind=kind, fill_value=fill_value, fraction=scale,
                                              level_scaling=level_scaling, max_shift=max_shift, max_levels=max_levels)
        else:
            frequencies = self.mf_amplitude.index
            self.generate_hilbert_params(c_levels_bi_scale=c_levels_bi_scale, scale=scale, max_shift=max_shift,
                                         c_levels_mono=c_levels_mono, c_levels_bi=c_levels_bi,
                                         alpha_0_mono=alpha_0_mono,
                                         alpha_0_bi=alpha_0_bi, fd_limits=fd_limits, kind=kind, method=method)
        # self.hilbert_params = generate_hilbert_params(self.mf_amplitude, c_levels_bi_scale=c_levels_bi_scale, scale=scale, fd_lower=fd_lower, fd_upper=fd_upper, max_shift=max_shift)
        # self.generate_hilbert_params(c_levels_bi_scale=c_levels_bi_scale, scale=scale)



        a_array = np.zeros(frequencies.shape[0], dtype=complex)

        params = deepcopy(self.parameters)
        for fd_idx, fd, alpha0, c_levels in tqdm(
                zip(np.arange(self.hilbert_params.index.shape[0]), self.hilbert_params.index,
                    self.hilbert_params['alpha_0'], self.hilbert_params['c_levels'])):
            params.fd = fd
            params.c_levels = c_levels
            ham = ham_gen_jc(params, alpha=alpha0)
            c_ops = c_ops_gen_jc(params, alpha=alpha0)
            try:
                rho = steadystate(ham, c_ops, **solver_kwargs)
                a = tensor(qeye(2), destroy(params.c_levels)) + alpha0
                a_array[fd_idx] = expect(rho, a)
            except:
                print('Failure at fd = ' + str(fd))
                a_array[fd_idx] = np.nan

        self.me_amplitude = pd.DataFrame(a_array, index=frequencies)

    def plot(self, axes=None, mf=True, me=True, db=True, me_kwargs={'marker': 'o'}, mf_kwargs={'marker': 'o'}):

        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.set_xlabel(r'$f_d$ (GHz)')
            axes.set_ylabel(r'|$\langle a \rangle$|')

        if db:
            if me:
                if self.me_amplitude is not None:
                    axes.plot(self.me_amplitude.dropna().index, 20 * np.log10(np.abs(self.me_amplitude.dropna())),
                              **me_kwargs)
            if mf:
                if self.mf_amplitude.shape[1] == 1:
                    axes.plot(self.mf_amplitude.index, 20 * np.log10(np.abs(self.mf_amplitude['a'])), **mf_kwargs)
                else:
                    axes.plot(self.mf_amplitude.index, 20 * np.log10(np.abs(self.mf_amplitude['a_bright'])),
                              **mf_kwargs)
                    axes.plot(self.mf_amplitude.index, 20 * np.log10(np.abs(self.mf_amplitude['a_dim'])), **mf_kwargs)
        else:
            if me:
                if self.me_amplitude is not None:
                    axes.plot(self.me_amplitude.dropna().index, np.abs(self.me_amplitude.dropna()), **me_kwargs)
            if mf:
                if self.mf_amplitude.shape[1] == 1:
                    axes.plot(self.mf_amplitude.index, np.abs(self.mf_amplitude['a']), **mf_kwargs)
                else:
                    axes.plot(self.mf_amplitude.index, np.abs(self.mf_amplitude['a_bright']), **mf_kwargs)
                    axes.plot(self.mf_amplitude.index, np.abs(self.mf_amplitude['a_dim']), **mf_kwargs)

    def plot_transmission(self, axes=None, scale=4.851024710399999e-09, exp=True, sim=True, me_kwargs={'marker': 'o'},
                          mf_kwargs={'marker': 'o'}):

        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.set_ylabel(r'$T_{NA}$ (dB)')
            axes.set_xlabel(r'$f_{df}$ (GHz)')

        if sim and self.me_amplitude is not None:
            self.transmission = scale * np.abs(self.me_amplitude.dropna()) ** 2 / self.parameters.eps ** 2
            axes.plot(self.transmission.index, 10 * np.log10(self.transmission), label='Sim', **me_kwargs)

        if exp and self.transmission_exp is not None:
            axes.plot(self.transmission_exp.index, self.transmission_exp, label='Exp')

    def load_exp(self, path):
        self.transmission_exp = pd.read_csv(path, dtype=float, header=None).T
        self.transmission_exp = self.transmission_exp.set_index(0)


def generate_hilbert_params(mf_amplitude, fd_limits=None, scale=0.5, c_levels_mono=10, c_levels_bi=10, alpha_0_mono=0, alpha_0_bi=0,
                            c_levels_bi_scale=1.0, max_shift=True, kind='linear', method='extrapolate_alpha_0'):
    if 'a_dim' not in mf_amplitude.columns:

        hilbert_params = deepcopy(mf_amplitude)
        hilbert_params.columns = ['alpha_0']
        hilbert_params['c_levels'] = c_levels_mono

    elif method is 'static':
        n_frequencies = mf_amplitude.shape[0]
        hilbert_params = pd.DataFrame(alpha_0_mono*np.ones([n_frequencies,1],dtype=complex), columns=['alpha_0'], index=mf_amplitude.index)
        hilbert_params['c_levels'] = c_levels_mono
        if fd_limits is not None:
            hilbert_params['c_levels'][fd_limits[0]:fd_limits[1]] = c_levels_bi
            hilbert_params['alpha_0'][fd_limits[0]:fd_limits[1]] = alpha_0_bi

    else:

        mf_amplitude_bistable = mf_amplitude.dropna()
        bistable_frequencies = mf_amplitude_bistable.index

        alpha_diff_bistable = mf_amplitude_bistable['a_bright'] - mf_amplitude_bistable['a_dim']
        alpha_diff_bistable_min = np.min(np.abs(alpha_diff_bistable))
        alpha_dim_bistable = mf_amplitude_bistable['a_dim']

        if max_shift:
            alpha_diff_bistable_unit = alpha_diff_bistable / np.abs(alpha_diff_bistable)
            alpha_0_bistable = alpha_dim_bistable + scale * alpha_diff_bistable_min * alpha_diff_bistable_unit
        else:
            alpha_0_bistable = alpha_dim_bistable + scale * alpha_diff_bistable

        if fd_limits is not None:

            if method not in ['extrapolate_alpha_0', 'extrapolate_diff']:
                raise Exception('Method not recognised.')

            bistable_frequencies = mf_amplitude[fd_limits[0]:fd_limits[1]].index

            if method is 'extrapolate_alpha_0':

                alpha_0_bistable_re_func = interp1d(alpha_0_bistable.index, alpha_0_bistable.values.real,
                                                    fill_value='extrapolate', kind=kind)
                alpha_0_bistable_im_func = interp1d(alpha_0_bistable.index, alpha_0_bistable.values.imag,
                                                    fill_value='extrapolate', kind=kind)

                def alpha_0_bistable_func_single(fd):
                    return alpha_0_bistable_re_func(fd) + 1j * alpha_0_bistable_im_func(fd)

                alpha_0_bistable_func = np.vectorize(alpha_0_bistable_func_single, otypes=[complex])
                alpha_0_bistable = alpha_0_bistable_func(bistable_frequencies)
                alpha_0_bistable = pd.Series(alpha_0_bistable, index=bistable_frequencies)

            elif method is 'extrapolate_diff':

                diff_re_func = interp1d(alpha_diff_bistable.index, alpha_diff_bistable.values.real,
                                        fill_value='extrapolate', kind=kind)
                diff_im_func = interp1d(alpha_diff_bistable.index, alpha_diff_bistable.values.imag,
                                        fill_value='extrapolate', kind=kind)

                def diff_func_single(fd):
                    return diff_re_func(fd) + 1j * diff_im_func(fd)

                diff_func = np.vectorize(diff_func_single, otypes=[complex])

                upper_mf_bistable_fd = mf_amplitude.dropna().index[-1]

                if fd_limits[1] < upper_mf_bistable_fd or fd_limits[0] > upper_mf_bistable_fd:
                    raise Exception('Frequency range does not cover the upper bistability crossover.')

                lower_midpoint_frequencies = mf_amplitude[fd_limits[0]:upper_mf_bistable_fd].index
                diff_lower = diff_func(lower_midpoint_frequencies)
                diff_lower_unit = diff_lower / np.abs(diff_lower)
                alpha_dim_lower = mf_amplitude['a_dim'][lower_midpoint_frequencies]
                alpha_0_lower = alpha_dim_lower + scale * alpha_diff_bistable_min * diff_lower_unit
                alpha_0_lower = pd.Series(alpha_0_lower, index=lower_midpoint_frequencies)

                upper_midpoint_frequencies = mf_amplitude[upper_mf_bistable_fd:fd_limits[1]].index[1:]
                diff_upper = diff_func(upper_midpoint_frequencies)
                diff_upper_unit = diff_upper / np.abs(diff_upper)
                alpha_bright_upper = mf_amplitude['a_bright'][upper_midpoint_frequencies]
                alpha_0_upper = alpha_bright_upper + (scale - 1) * alpha_diff_bistable_min * diff_upper_unit
                alpha_0_upper = pd.Series(alpha_0_upper, index=upper_midpoint_frequencies)

                alpha_0_bistable = pd.concat([alpha_0_lower, alpha_0_upper])

        fd_lower = alpha_0_bistable.index[0]
        fd_upper = alpha_0_bistable.index[-1]
        alpha_0_monostable_bright = mf_amplitude['a_bright'].dropna().loc[fd_upper:]
        alpha_0_monostable_bright = alpha_0_monostable_bright.iloc[1:]
        alpha_0_monostable_dim = mf_amplitude['a_dim'].dropna().loc[:fd_lower]
        alpha_0_monostable_dim = alpha_0_monostable_dim.iloc[:-1]

        hilbert_params_mono = pd.concat(
            [alpha_0_monostable_bright.to_frame('alpha_0'), alpha_0_monostable_dim.to_frame('alpha_0')])
        hilbert_params_mono['c_levels'] = c_levels_mono

        hilbert_params_bi = alpha_0_bistable.to_frame('alpha_0')
        hilbert_params_bi['c_levels'] = int(np.ceil(c_levels_bi_scale * alpha_diff_bistable_min ** 2))

        hilbert_params = pd.concat([hilbert_params_mono, hilbert_params_bi])
        hilbert_params = hilbert_params.sort_index()

    return hilbert_params


class HysteresisSweep:
    def __init__(self, params, fd_array, t_end, initial_direction='up'):
        self.params = params
        self.fd_array = fd_array
        self.sweeps = None
        self.t_end = t_end
        self.previous_state = tensor(basis(2, 0), basis(self.params.c_levels, 0))
        self.next_direction = initial_direction
        self.next_sweep_idx = 0

    def gen_sweep(self):
        sm = tensor(sigmam(), qeye(self.params.c_levels))
        a = tensor(qeye(2), destroy(self.params.c_levels))
        e_ops = [a.dag() * a, sm.dag() * sm, a, sm]
        options = Options(store_states=True)
        times = np.linspace(0, self.t_end, 1001)
        collected_results_frame = None
        params = deepcopy(self.params)
        if self.next_direction is 'up':
            next_fd_array = self.fd_array
        else:
            next_fd_array = np.flip(self.fd_array)
        for fd in next_fd_array:
            params.fd = fd
            ham = ham_gen_jc(params)
            c_ops = c_ops_gen_jc(params)
            result = mcsolve(ham, self.previous_state, times, c_ops, e_ops, ntraj=1, options=options)
            result_frame = pd.DataFrame(np.array(result.expect).T, columns=['n_c', 'n_t', 'a', 'b'])
            result_frame['fd'] = params.fd
            result_frame['t'] = times
            result_frame['direction'] = self.next_direction
            result_frame['count'] = self.next_sweep_idx / 2
            result_frame.set_index(['direction', 'count', 'fd', 't'], inplace=True)
            dtype_dict = dict()
            dtype_dict['a'] = 'complex'
            dtype_dict['b'] = 'complex'
            dtype_dict['n_c'] = 'float'
            dtype_dict['n_t'] = 'float'
            result_frame = result_frame.astype(dtype_dict)
            if collected_results_frame is None:
                collected_results_frame = result_frame
            else:
                collected_results_frame = pd.concat([collected_results_frame, result_frame])
            self.previous_state = result.states[0, -1]
        if self.sweeps is None:
            self.sweeps = collected_results_frame
        else:
            self.sweeps = pd.concat([collected_results_frame, self.sweeps])

        if self.next_direction == 'up':
            self.next_direction = 'down'
        else:
            self.next_direction = 'up'
        self.next_sweep_idx += 1