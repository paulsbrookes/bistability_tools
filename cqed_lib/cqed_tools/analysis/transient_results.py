import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from .loading import load_settings
from .fitting import decay_gen


class TransientResults:
    def __init__(self, directory):
        self.defect_list = []
        self.exp_results = None
        self.fit_params = None
        self.ss_results = None
        self.axes = None
        self.fit = None
        self.list = []
        walk = os.walk(directory)
        for path_info in walk:
            path = path_info[0]
            if os.path.exists(path + '/ss_results.csv'):
                settings = load_settings(path + '/settings.csv')
                ss_results = pd.read_csv(path + '/ss_results.csv', index_col=0)
                ss_results['eps'] = settings.eps
                ss_results['fd'] = settings.fd
                ss_results.set_index(['eps', 'fd'], append=True, inplace=True)
                self.list.append(ss_results)
                if self.ss_results is None:
                    self.ss_results = ss_results
                else:
                    self.ss_results = pd.concat([self.ss_results, ss_results], sort=True)
        self.ss_results['a_exp'] = self.ss_results['a_op_re'] + 1j * self.ss_results['a_op_im']
        self.ss_results['b_exp'] = self.ss_results['sm_op_re'] + 1j * self.ss_results['sm_op_im']
        self.ss_results.sort_index(inplace=True)

    def plot_transmission(self, axes=None, label=True):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.set_xlabel(r'$f_d$ (GHz)')
        axes.set_ylabel(r'$|\langle a \rangle|$')
        eps_level_idx = self.ss_results.index.names.index('eps')
        eps_array = self.ss_results.index.levels[eps_level_idx]
        for eps in eps_array:
            cut = self.ss_results.xs(eps, level='eps')
            fd_array = cut.index.get_level_values('fd')
            transmission_array = np.abs(cut['a_exp'])
            index_array = cut.index.get_level_values('job_index')
            axes.plot(fd_array, transmission_array, marker='o')
            if label:
                for index, fd, trans in zip(index_array, fd_array, transmission_array):
                    axes.annotate(index, (fd, trans))

    def plot_slowdown(self, axes=None, eps_indices=None, label=True, legend=False, interpolate=True, kwargs=dict()):
        if self.fit_params is None:
            self.slowdown_calc()
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        self.fit_params_pruned = self.fit_params.drop(labels=self.defect_list, level='job_index')
        self.fit_params_pruned.sort_index(level='fd', inplace=True)
        axes.set_xlabel(r'$f_d$ (GHz)')
        axes.set_ylabel(r'$T_s$ ($\mu$s)')
        eps_level_idx = self.ss_results.index.names.index('eps')
        eps_array = self.ss_results.index.levels[eps_level_idx]
        if eps_indices is not None:
            eps_array = eps_array[eps_indices]
        for eps in eps_array:
            cut = self.fit_params_pruned.xs(eps, level='eps')
            cut = cut.reset_index()
            cut = cut.drop_duplicates(subset='fd')
            fd_array = cut['fd'].values
            slowdown_array = np.abs(cut['Ts'])
            if legend:
                legend_label = eps
            else:
                legend_label = ''
            if interpolate:
                new_fd_array = np.linspace(fd_array[0], fd_array[-1], 201)
                interp_func = interp1d(fd_array, slowdown_array, kind='cubic')
                axes.plot(new_fd_array, interp_func(new_fd_array), label=legend_label, **kwargs)
            else:
                axes.plot(fd_array, slowdown_array, label=legend_label, **kwargs)
                if label:
                    index_array = cut['job_index'].values
                    for index, fd, Ts in zip(index_array, fd_array, slowdown_array):
                        axes.annotate(index, (fd, Ts))

    def plot_slowdown_exp(self, axes=None, threshold=10.0, powers=None, errors=True, legend_label=True, kwargs=dict()):
        if axes is None:
            if self.axes is None:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                self.axes = axes
            else:
                axes = self.axes
        mask = self.exp_results['error'] < threshold
        self.exp_results_pruned = self.exp_results[mask]
        if powers is None:
            powers = self.exp_results_pruned.index.levels[0]
        for power in powers:
            cut = self.exp_results_pruned.xs(power, level=0, drop_level=False)
            if legend_label:
                label = power
            else:
                label = ''
            if errors:
                axes.errorbar(cut.index.get_level_values('fd'), cut['Ts'].values, yerr=cut['error'], label=label,
                              **kwargs)
            else:
                axes.plot(cut.index.get_level_values('fd'), cut['Ts'].values, label=label, **kwargs)

    def load_transients(self, directory):
        self.transients = None
        self.slowdown = None
        walk = os.walk(directory)
        for path_info in walk:
            path = path_info[0]
            if os.path.exists(path + '/results.csv'):
                settings = load_settings(path + '/settings.csv')
                results = pd.read_csv(path + '/results.csv', index_col=0)
                results.index /= (2 * np.pi * 1000)
                results['job_index'] = settings.job_index
                results['eps'] = settings.eps
                results['fd'] = settings.fd
                results.set_index(['job_index', 'eps', 'fd'], append=True, inplace=True)
                if self.transients is None:
                    self.transients = results
                else:
                    self.transients = pd.concat([self.transients, results], sort=True)
        self.transients = self.transients.reorder_levels(['job_index', 'eps', 'fd', 'times'])
        self.transients.sort_index(inplace=True)

    def plot_transient(self, job_index, y_quantity='a_op_re', axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.set_xlabel(r'Time ($\mu$s)')
        transient = self.transients.xs(job_index, level='job_index')
        axes.plot(transient.index.get_level_values('times'), transient[y_quantity])

    def plot_fit(self, job_index, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.set_xlabel(r'Time ($\mu$s)')
        transient = self.transients.xs(job_index, level='job_index')
        times = transient.index.get_level_values('times')
        axes.plot(times, transient['a_op_re'])
        if job_index not in self.fit_params.index.get_level_values('job_index'):
            self.fit_transient(job_index)
        a_op_re = self.ss_results['a_op_re'].xs(job_index, level='job_index').values[0]
        decay_func = decay_gen(a_op_re)
        popt = self.fit_params.xs(job_index, level='job_index').values[0, :]
        axes.plot(times, decay_func(times, *popt))

    def fit_transient(self, transient_index, t0=3.0):
        package_index = self.ss_results.xs(transient_index, level=self.ss_results.index.names).index
        i_signal = self.transients['a_op_re'].loc[transient_index]
        t_end = i_signal.index.get_level_values('times')[-1]
        i_sample = i_signal.loc[t0:t_end]
        if i_sample.shape[0] > 1:
            times = i_sample.index.get_level_values('times')
            a_ss_re = self.ss_results.xs(transient_index, level=self.ss_results.index.names)['a_op_re'].values[0]
            decay_fixed = decay_gen(a_ss_re)
            d_est = (i_sample.iloc[-1] - i_sample.iloc[0]) / (times[-1] - times[0])
            T_est = (a_ss_re - i_sample.iloc[0]) / d_est
            A_est = -d_est * T_est * np.exp(times[0] / T_est)
            popt, pcov = curve_fit(f=decay_fixed, xdata=times, ydata=i_sample.values, p0=[A_est, T_est])
            fit_params = pd.DataFrame(np.array([popt]), index=package_index, columns=['A', 'Ts'])
            if self.fit_params is None:
                self.fit_params = fit_params
            else:
                self.fit_params = pd.concat([self.fit_params, fit_params], sort=True)
            self.fit_params.sort_index(inplace=True)

    def slowdown_calc(self):
        transient_indices = self.transients.index.droplevel('times')
        steadystate_indices = self.ss_results.index
        for index in set(transient_indices).intersection(steadystate_indices):
            try:
                self.fit_transient(index)
            except:
                print(index)

    def load_exp(self, path):
        self.exp_results = pd.read_hdf(path)

    def load_calibration(self, path):
        self.calibration = pd.read_hdf(path)

    def load_exp_spectra(self, path):
        self.exp_spectra = pd.read_hdf(path)