from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os
from qutip import *


def lorentzian_func(f, A, f_r, Q, c):
    return A*(f_r/Q)/((f_r/Q)**2 + 4*(f-f_r)**2)**0.5 + c


def lin_func(x, a, b):
    return a*x + b


def quadratic_func(x, a, b, c):
    return a*(x-b)**2 + c


def lorentzian_fit(spectrum):
    max_idx = np.argmax(spectrum).values[()]
    A_est = spectrum[max_idx]
    Q_est = 10000
    try:
        f_r_est = spectrum.frequency[max_idx]
        popt, pcov = curve_fit(lorentzian_func, spectrum.frequency, spectrum.values, p0=[A_est, f_r_est, Q_est, 0.01])
    except:
        f_r_est = spectrum.f_d[max_idx]
        popt, pcov = curve_fit(lorentzian_func, spectrum.f_d, spectrum.values, p0=[A_est, f_r_est, Q_est, 0.01])
    return popt, pcov


def Q_calc(spectrum, display=False):
    popt, pcov = lorentzian_fit(spectrum)
    Q_factor = popt[2]

    if display:
        fig, axes = plt.subplots(1,1)
        spectrum.plot(ax=axes)
        try:
            axes.plot(spectrum.frequency, lorentzian_func(spectrum.frequency, *popt), 'g--')
        except:
            axes.plot(spectrum.f_d, lorentzian_func(spectrum.f_d, *popt), 'g--')

    return Q_factor, np.sqrt(pcov[2,2])


def res_finder_row(row):
    mid_index = row.argmax().values[()]
    lower = mid_index - 5
    upper = mid_index + 5
    window_frequencies = row.frequency[lower:upper].values[()]
    window_values = row[lower:upper].values[()]
    p = np.polyfit(window_frequencies, window_values, 2)
    resonance = -p[1]/(2*p[0])
    value = p[0]*resonance**2 + p[1]*resonance + p[2]
    return resonance, value


def res_finder(array):
    n_powers = array.shape[0]
    frequencies = []
    values = []
    for i in range(n_powers):
        row = array[i,:].dropna('frequency')
        resonance, value = res_finder_row_alt(row)
        frequencies.append(resonance)
        values.append(value)
    return array.power, frequencies, values


def res_finder_row_alt(row):
    values = row.values
    frequencies = row.frequency
    N = 5
    clip = (N - 1) / 2
    running_mean = np.convolve(values, np.ones((N,)) / N, mode='valid')
    running_frequencies = frequencies[clip:-clip]

    mid_index = running_mean.argmax()
    lower = mid_index - 5
    upper = mid_index + 5
    window_frequencies = running_frequencies[lower:upper]
    window_values = running_mean[lower:upper]

    p = np.polyfit(window_frequencies, window_values, 2)
    resonance = -p[1] / (2 * p[0])
    value = p[0] * resonance ** 2 + p[1] * resonance + p[2]
    return resonance, value


def eps_to_power(eps):
    power = freq_to_power(eps_to_freq(eps))
    return power


def power_to_eps(power):
    eps = freq_to_eps(power_to_freq(power))
    return eps


def lin_func_single(x, a):
    return 20*x/np.log(10) + a


def epsilon_calc(power, a, b):
    eps = np.exp((power - b)/a)
    return eps


def load_experimental(path, display=True):
    dataframe = pd.read_csv(path, skiprows=5, sep='\t')
    index = pd.read_csv(path, skiprows=3, sep='\t', nrows=1, header=None)
    index = index.T[0].astype(np.float)
    dataframe.index = index.values
    dataframe = dataframe.T
    dataframe.index = dataframe.index.astype(np.float)*1e-9 + 0.125
    dataframe = dataframe.iloc[:, ::-1]
    if display:
        fig, axes = plt.subplots(1,1)
        dataframe.iloc[:,:].plot(ax=axes)
    return dataframe


def peak_finder(dataframe, N=1, display=False, interpolate=True):
    if display:
        fig, axes = plt.subplots(1,1)
    n_powers = dataframe.shape[1]
    peak_indices = []
    peak_values = []
    for idx in range(n_powers):
        cut = dataframe.iloc[:,idx]
        cut = cut.dropna()
        smoothed_index = np.convolve(cut.index, np.ones((N,))/N, mode='valid')
        smoothed_data = np.convolve(cut.values, np.ones((N,))/N, mode='valid')
        if interpolate:
            interp_func = interp1d(smoothed_index, smoothed_data, kind='cubic')
            fine_indices = np.linspace(smoothed_index[0],smoothed_index[-1],2001)
            fine_values = interp_func(fine_indices)
        else:
            fine_indices = smoothed_index
            fine_values = smoothed_data
        max_idx = np.argmax(fine_values)
        peak_indices.append(fine_indices[max_idx])
        peak_values.append(fine_values[max_idx])
        if display:
            #cut.plot(ax=axes)
            axes.plot(fine_indices,fine_values)
    peak_array = np.array([peak_indices, peak_values]).T
    peak_frame = pd.DataFrame(peak_array, index=dataframe.columns, columns=['frequency','value'])
    if display:
        axes.scatter(peak_frame['frequency'],peak_frame['value'])
    return peak_frame


def load_simulated(sweep_path, display=True):
    fig, axes = plt.subplots(1, 1)

    bottom_levels = []
    for idx, (dirpath, dirnames, filenames) in enumerate(os.walk(sweep_path)):
        if not dirnames:
            bottom_levels.append(dirpath)

    transmissions = np.array([])
    frequencies = np.array([])
    drives = np.array([])

    spectrum = None
    legend_content = []
    collected_results = None
    cut_list = []

    for bottom_path in bottom_levels:
        results = pd.read_hdf(bottom_path + '/results.h5')
        params = qload(bottom_path + '/params')
        results = results.set_index('fd_points')
        a_abs = np.abs(results['a'])
        a_abs.plot(ax=axes)
        marker = str(1000 * params.eps)
        legend_content.append(marker)
        results['epsilon'] = params.eps
        if collected_results is None:
            collected_results = results
        else:
            collected_results = pd.concat([collected_results, results])
        a_abs.name = params.eps
        cut_list.append(a_abs)

    collected_results.set_index('epsilon', append=True, inplace=True)

    simulated_results = pd.concat(cut_list, axis=1)
    return simulated_results
    peak_frame = peak_finder(simulated_results, N=1)
    eps_array = simulated_results.columns

    return peak_frame
    freq_array = peak_frame['frequency']

    axes.scatter(freq_array, peak_frame['values'])

    legend = axes.legend(legend_content)
    legend.set_title(r'$\epsilon$ / MHz')

    return simulated_results


def load_simulated_frame(sweep_path):
    bottom_levels = []
    for idx, (dirpath, dirnames, filenames) in enumerate(os.walk(sweep_path)):
        if not dirnames:
            bottom_levels.append(dirpath)

    collected_results = None

    for bottom_path in bottom_levels:
        results = pd.read_hdf(bottom_path + '/results.h5')
        params = qload(bottom_path + '/params')
        results = results.set_index('fd_points')
        results['epsilon'] = params.eps
        results = results.reset_index().set_index(['epsilon', 'fd_points'])
        if collected_results is None:
            collected_results = [results]
        else:
            collected_results.append(results)

    collected_results = collected_results[0].append(collected_results[1:])
    collected_results.index.names = ['eps', 'fd']
    collected_results.sort_index(inplace=True)

    return collected_results