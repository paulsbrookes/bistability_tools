from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


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