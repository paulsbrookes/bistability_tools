import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
from scipy.special import factorial
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from qutip.ui.progressbar import TextProgressBar
import os
import time
import pandas as pd
import xarray as xr
from collections import OrderedDict
from datetime import datetime
import scipy.special as special
import scipy
import scipy.sparse.linalg as lin


class SpectrumOptions:
    def __init__(self, fd_lower=10.46, fd_upper=10.52, threshold=0.05, display=False):
        self.fd_lower = fd_lower
        self.fd_upper = fd_upper
        self.threshold = threshold
        self.display = display


def lin_func(x, a, b):
    return a * x + b


def steadystate_occupations_calc(params):
    c_ops = collapse_operators(params)
    H = hamiltonian(params)
    rho_ss = steadystate(H, c_ops)
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    n_t = expect(sm.dag() * sm, rho_ss)
    n_c = expect(a.dag() * a, rho_ss)
    return n_t, n_c


def quadratic_func(x, a, b, c):
    return a * (x - b) ** 2 + c


def lorentzian_func(f, A, f_r, Q, c):
    return A * (f_r / Q) / (((f_r / Q) ** 2 + 4 * (f - f_r) ** 2)) ** 0.5 + c


def lorentzian_fit(x, y):
    max_idx = np.argmax(y)
    A_est = y[max_idx]
    Q_est = 10000
    f_r_est = x[max_idx]
    popt, pcov = curve_fit(lorentzian_func, x, y, p0=[A_est, f_r_est, Q_est, 0.01])
    return popt, pcov


class Parameters:
    def __init__(self, fc, Ej, g, Ec, eps, fd, kappa, gamma, t_levels, c_levels, gamma_phi, kappa_phi, n_t, n_c):
        self.fc = fc
        self.Ej = Ej
        self.eps = eps
        self.g = g
        self.Ec = Ec
        self.gamma = gamma
        self.kappa = kappa
        self.t_levels = t_levels
        self.c_levels = c_levels
        self.fd = fd
        self.gamma_phi = gamma_phi
        self.kappa_phi = kappa_phi
        self.n_t = n_t
        self.n_c = n_c
        self.labels = ['f_d', 'eps', 'E_j', 'f_c', 'g', 'kappa', 'kappa_phi', 'gamma', 'gamma_phi', 'E_c', 'n_t', 'n_c']

    def copy(self):
        params = Parameters(self.fc, self.Ej, self.g, self.Ec, self.eps, self.fd, self.kappa, self.gamma, self.t_levels,
                            self.c_levels, self.gamma_phi, self.kappa_phi, self.n_t, self.n_c)
        return params


def collapse_operators(params):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(params.kappa * (params.n_c + 1)) * a)
    c_ops.append(np.sqrt(params.kappa * params.n_c) * a.dag())
    c_ops.append(np.sqrt(params.gamma * (params.n_t + 1)) * sm)
    c_ops.append(np.sqrt(params.gamma * params.n_t) * sm.dag())
    # dispersion_op = dispersion_op_gen(params)
    dispersion_op = sm.dag() * sm
    c_ops.append(np.sqrt(params.gamma_phi) * dispersion_op)
    return c_ops


def charge_dispersion_calc(level, Ec, Ej):
    dispersion = (-1) ** level * Ec * 2 ** (4 * level + 5) * np.sqrt(2.0 / np.pi) * (Ej / (2 * Ec)) ** (
                level / 2.0 + 3 / 4.0) * np.exp(-np.sqrt(8 * Ej / Ec))
    dispersion /= factorial(level)
    return dispersion


def dispersion_op_gen(params):
    Ec = params.Ec
    Ej = params.Ej
    normalization = charge_dispersion_calc(1, Ec, Ej) - charge_dispersion_calc(0, Ec, Ej)
    dispersion_op = 0
    for i in range(params.t_levels):
        dispersion_op += fock_dm(params.t_levels, i) * charge_dispersion_calc(i, Ec, Ej)
    dispersion_op /= normalization
    dispersion_op = tensor(qeye(params.c_levels), dispersion_op)
    return dispersion_op


def local_maxima(array):
    truth_array = np.r_[True, array[1:] > array[:-1]] & np.r_[array[:-1] > array[1:], True]
    indices = np.argwhere(truth_array)[:, 0]
    return indices


class Results:
    def __init__(self, params=np.array([]), fd_points=np.array([]),
                 transmissions=np.array([]), edge_occupations_c=np.array([]), edge_occupations_t=np.array([])):
        self.params = params
        self.fd_points = fd_points
        self.transmissions = transmissions
        self.edge_occupations_c = edge_occupations_c
        self.edge_occupations_t = edge_occupations_t
        self.abs_transmissions = np.absolute(self.transmissions)
        self.size = self.fd_points.size

    def concatenate(self, results):
        combined_params = np.concatenate([self.params, results.params])
        combined_fd_points = np.concatenate([self.fd_points, results.fd_points])
        combined_transmissions = np.concatenate([self.transmissions, results.transmissions])
        combined_edge_occupations_c = np.concatenate([self.edge_occupations_c, results.edge_occupations_c])
        combined_edge_occupations_t = np.concatenate([self.edge_occupations_t, results.edge_occupations_t])
        sort_indices = np.argsort(combined_fd_points)
        combined_params = combined_params[sort_indices]
        combined_fd_points = combined_fd_points[sort_indices]
        combined_transmissions = combined_transmissions[sort_indices]
        combined_edge_occupations_c = combined_edge_occupations_c[sort_indices]
        combined_edge_occupations_t = combined_edge_occupations_t[sort_indices]
        combined_results = Results(combined_params, combined_fd_points,
                                   combined_transmissions, combined_edge_occupations_c, combined_edge_occupations_t)
        return combined_results

    def delete(self, indices):
        reduced_params = np.delete(self.params, indices)
        reduced_fd_points = np.delete(self.fd_points, indices)
        reduced_transmissions = np.delete(self.transmissions, indices)
        reduced_edge_occupations_c = np.delete(self.edge_occupations_c, indices)
        reduced_edge_occupations_t = np.delete(self.edge_occupations_t, indices)
        reduced_results = Results(reduced_params, reduced_fd_points,
                                  reduced_transmissions, reduced_edge_occupations_c, reduced_edge_occupations_t)
        params_change = (reduced_params == self.params)
        fd_points_change = (reduced_fd_points == self.fd_points)
        transmissions_change = (reduced_transmissions == self.transmissions)
        edge_occupations_c_change = (reduced_edge_occupations_c == self.edge_occupations_c)
        edge_occupations_t_change = (reduced_edge_occupations_t == self.edge_occupations_t)
        print np.all([params_change, fd_points_change, transmissions_change, edge_occupations_c_change,
                      edge_occupations_t_change])
        return reduced_results

    def queue(self):
        queue = Queue(self.params, self.fd_points)
        return queue


class Queue:
    def __init__(self, params=np.array([]), fd_points=np.array([])):
        self.params = params
        self.fd_points = fd_points
        self.size = self.fd_points.size
        sort_indices = np.argsort(self.fd_points)
        self.fd_points = self.fd_points[sort_indices]
        self.params = self.params[sort_indices]

    def curvature_generate(self, results, threshold=0.05):
        curvature_info = CurvatureInfo(results, threshold)
        self.fd_points = curvature_info.new_points()
        self.params = hilbert_interpolation(self.fd_points, results)
        self.size = self.fd_points.size
        sort_indices = np.argsort(self.fd_points)
        self.fd_points = self.fd_points[sort_indices]
        self.params = self.params[sort_indices]

    def hilbert_generate(self, results, threshold_c, threshold_t):
        suggested_c_levels = []
        suggested_t_levels = []
        overload_occurred = False
        for index, params_instance in enumerate(results.params):
            threshold_c_weighted = threshold_c / params_instance.c_levels
            threshold_t_weighted = threshold_t / params_instance.t_levels
            overload_c = (results.edge_occupations_c[index] > threshold_c_weighted)
            overload_t = (results.edge_occupations_t[index] > threshold_t_weighted)
            if overload_c:
                overload_occurred = True
                suggestion = size_correction(
                    results.edge_occupations_c[index], params_instance.c_levels, threshold_c_weighted / 2)
            else:
                suggestion = params_instance.c_levels
            suggested_c_levels.append(suggestion)
            if overload_t:
                overload_occurred = True
                suggestion = size_correction(
                    results.edge_occupations_t[index], params_instance.t_levels, threshold_t_weighted / 2)
            else:
                suggestion = params_instance.t_levels
            suggested_t_levels.append(suggestion)
        if overload_occurred:
            c_levels_new = np.max(suggested_c_levels)
            t_levels_new = np.max(suggested_t_levels)
            self.fd_points = results.fd_points
            for index, params_instance in enumerate(results.params):
                results.params[index].t_levels = t_levels_new
                results.params[index].c_levels = c_levels_new
            self.params = results.params
            self.size = results.size
            return Results()
        else:
            self.fd_points = np.array([])
            self.params = np.array([])
            self.size = 0
            return results

    def hilbert_generate_alternate(self, results, threshold_c, threshold_t):
        old_c_levels = np.zeros(results.size)
        suggested_c_levels = np.zeros(results.size)
        old_t_levels = np.zeros(results.size)
        suggested_t_levels = np.zeros(results.size)
        for index, params_instance in enumerate(results.params):
            suggested_c_levels[index] = \
                size_suggestion(results.edge_occupations_c[index], params_instance.c_levels, threshold_c)
            old_c_levels[index] = params_instance.c_levels
            suggested_t_levels[index] = \
                size_suggestion(results.edge_occupations_t[index], params_instance.t_levels, threshold_t)
            old_t_levels[index] = params_instance.t_levels
        if np.any(suggested_c_levels > old_c_levels) or np.any(suggested_t_levels > old_t_levels):
            c_levels_new = np.max(suggested_c_levels)
            t_levels_new = np.max(suggested_t_levels)
            self.fd_points = results.fd_points
            for index, params_instance in enumerate(results.params):
                results.params[index].t_levels = t_levels_new
                results.params[index].c_levels = c_levels_new
            self.params = results.params
            self.size = results.size
            return Results()
        else:
            self.fd_points = np.array([])
            self.params = np.array([])
            self.size = 0
            return results


class CurvatureInfo:
    def __init__(self, results, threshold=0.05):
        self.threshold = threshold
        self.fd_points = results['fd_points'].values
        self.new_fd_points_unique = None
        self.abs_transmissions = np.abs(results['transmissions'].values)
        self.n_points = self.abs_transmissions.size

    def new_points(self):
        self.curvature_positions, self.curvatures = derivative(self.fd_points, self.abs_transmissions, 2)
        self.abs_curvatures = np.absolute(self.curvatures)
        self.mean_curvatures = moving_average(self.abs_curvatures, 2)
        self.midpoint_curvatures = \
            np.concatenate((np.array([self.abs_curvatures[0]]), self.mean_curvatures))
        self.midpoint_curvatures = \
            np.concatenate((self.midpoint_curvatures, np.array([self.abs_curvatures[self.n_points - 3]])))
        self.midpoint_transmissions = moving_average(self.abs_transmissions, 2)
        self.midpoint_curvatures_normed = self.midpoint_curvatures / self.midpoint_transmissions
        self.midpoints = moving_average(self.fd_points, 2)
        self.intervals = np.diff(self.fd_points)
        self.num_of_sections_required = \
            np.ceil(self.intervals * np.sqrt(self.midpoint_curvatures_normed / self.threshold))
        mask = self.num_of_sections_required > 0
        self.num_of_sections_required *= mask
        new_fd_points = np.array([])
        for index in np.arange(self.n_points - 1):
            multi_section = \
                np.linspace(self.fd_points[index], self.fd_points[index + 1], self.num_of_sections_required[index] + 1)
            new_fd_points = np.concatenate((new_fd_points, multi_section))
        unique_set = set(new_fd_points) - set(self.fd_points)
        self.new_fd_points_unique = np.array(list(unique_set))
        return self.new_fd_points_unique


def size_suggestion(edge_occupation, size, threshold):
    beta = fsolve(zero_func, 1, args=(edge_occupation, size - 1, size))
    new_size = - np.log(threshold) / beta
    new_size = int(np.ceil(new_size))
    return new_size


def size_correction(edge_occupation, size, threshold):
    beta_estimate = np.log(1 + 1 / edge_occupation) / size
    beta = fsolve(zero_func, beta_estimate, args=(edge_occupation, size - 1, size))
    new_size = 1 + np.log((1 - np.exp(-beta)) / threshold) / beta
    new_size = int(np.ceil(new_size))
    return new_size


def exponential_occupation(n, beta, size):
    factor = np.exp(-beta)
    f = np.power(factor, n) * (1 - factor) / (1 - np.power(factor, size))
    return f


def zero_func(beta, p, level, size):
    f = exponential_occupation(level, beta, size)
    f = f - p
    return f


def hilbert_interpolation(new_fd_points, results):
    c_levels_array = np.array([params.c_levels for params in results['params']])
    t_levels_array = np.array([params.t_levels for params in results['params']])
    fd_points = results.fd_points
    c_interp = interp1d(fd_points, c_levels_array)
    t_interp = interp1d(fd_points, t_levels_array)
    base_params = results['params'].iloc[0]
    params_list = []
    for fd in new_fd_points:
        new_params = base_params.copy()
        new_params.c_levels = int(round(c_interp(fd)))
        new_params.t_levels = int(round(t_interp(fd)))
        params_list.append(new_params)
    params_array = np.array(params_list)
    return params_array


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    averages = np.convolve(interval, window, 'same')
    return averages[window_size - 1: averages.size]


def derivative(x, y, n_derivative=1):
    derivatives = np.zeros(y.size - 1)
    positions = np.zeros(x.size - 1)
    for index in np.arange(y.size - 1):
        grad = (y[index + 1] - y[index]) / (x[index + 1] - x[index])
        position = np.mean([x[index], x[index + 1]])
        derivatives[index] = grad
        positions[index] = position

    if n_derivative > 1:
        positions, derivatives = derivative(positions, derivatives, n_derivative - 1)
    return positions, derivatives


def transmon_energies_calc(params):
    Ec = params.Ec
    Ej = params.Ej
    q = -Ej / (2 * Ec)
    n_levels = params.t_levels
    n_even = n_levels // 2 + n_levels % 2
    n_odd = n_levels // 2
    even_levels = np.arange(0, 2 * n_even, 2)
    odd_levels = np.arange(2, 2 * (n_odd + 1), 2)
    even_energies = Ec * special.mathieu_a(even_levels, q)
    odd_energies = Ec * special.mathieu_b(odd_levels, q)
    energies = np.hstack([even_energies, odd_energies])
    sorted_energies = np.sort(energies)
    sorted_energies -= sorted_energies[0]
    return sorted_energies


def transmon_hamiltonian_gen(params):
    energies = transmon_energies_calc(params)
    transmon_hamiltonian = 0
    for n, energy in enumerate(energies):
        transmon_hamiltonian += (energy - n * params.fd) * fock_dm(params.t_levels, n)
    transmon_hamiltonian = tensor(qeye(params.c_levels), transmon_hamiltonian)
    return transmon_hamiltonian


def overlap_func(theta, i, j, params):
    q = -params.Ej / (2 * params.Ec)
    if i % 2 == 0:
        bra = np.conjugate(special.mathieu_cem(i, q, theta * 180 / np.pi)[0]) * np.sqrt(2 / np.pi)
    else:
        bra = np.conjugate(special.mathieu_sem(i + 1, q, theta * 180 / np.pi)[0]) * np.sqrt(2 / np.pi)
    if j % 2 == 0:
        ket = special.mathieu_cem(j, q, theta * 180 / np.pi)[1] * np.sqrt(2 / np.pi)
    else:
        ket = special.mathieu_sem(j + 1, q, theta * 180 / np.pi)[1] * np.sqrt(2 / np.pi)
    overlap_point = bra * ket

    return overlap_point


def coupling_calc(i, j, params):
    coupling, error = scipy.integrate.quad(overlap_func, 0, np.pi, args=(i, j, params))
    return np.abs(coupling)


coupling_vec_calc = np.vectorize(coupling_calc)


def coupling_hamiltonian_gen(params):
    lower_levels = np.arange(0, params.t_levels - 1)
    upper_levels = np.arange(1, params.t_levels)
    coupling_array = coupling_vec_calc(lower_levels, upper_levels, params)
    coupling_array = coupling_array / coupling_array[0]
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    down_transmon_transitions = 0
    for i, coupling in enumerate(coupling_array):
        down_transmon_transitions += coupling * basis(params.t_levels, i) * basis(params.t_levels, i + 1).dag()
    down_transmon_transitions = tensor(qeye(params.c_levels), down_transmon_transitions)
    down_transmon_transitions *= a.dag()
    coupling_hamiltonian = down_transmon_transitions + down_transmon_transitions.dag()
    coupling_hamiltonian *= params.g
    return coupling_hamiltonian


def hamiltonian(params):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    transmon_hamiltonian = transmon_hamiltonian_gen(params)
    coupling_hamiltonian = coupling_hamiltonian_gen(params)
    H = (params.fc - params.fd) * a.dag() * a + transmon_hamiltonian + coupling_hamiltonian + params.eps * (a + a.dag())
    return H


def transmission_calc_array(queue, results):
    args = []
    for index, value in enumerate(queue.fd_points):
        args.append([value, queue.params[index]])
    # steady_states = parallel_map(transmission_calc, args, num_cpus=1, progress_bar=TextProgressBar())
    steady_states = []
    for arg in args:
        steady_states.append(transmission_calc(arg, results))
    transmissions = np.array([steady_state[0] for steady_state in steady_states])
    edge_occupations_c = np.array([steady_state[1] for steady_state in steady_states])
    edge_occupations_c = np.absolute(edge_occupations_c)
    edge_occupations_t = np.array([steady_state[2] for steady_state in steady_states])
    edge_occupations_t = np.absolute(edge_occupations_t)
    states = np.array([steady_state[3] for steady_state in steady_states])
    # results = Results(queue.params, queue.fd_points, transmissions, edge_occupations_c, edge_occupations_t)
    # abs_transmissions = np.absolute(transmissions)
    results = pd.DataFrame(
        [queue.params, queue.fd_points, transmissions, edge_occupations_c, edge_occupations_t, states]).T
    results.columns = ['params', 'fd_points', 'transmissions', 'edge_occupations_c', 'edge_occupations_t', 'states']
    dtypes = {'params': object, 'fd_points': np.float64, 'transmissions': np.complex, 'edge_occupations_c': np.float64,
              'edge_occupations_t': np.float64, 'states': object}
    results = results.astype(dtypes)
    return results


def steadystate_custom(H, c_ops, initial):
    L = liouvillian(H, c_ops)

    data = L.data
    csc = data.tocsc()

    if initial is None:
        eigenvector = None
    else:
        eigenvector = operator_to_vector(initial).data.todense()

    values, vectors = lin.eigs(csc, k=5, sigma=0.0, v0=eigenvector)
    sort_indices = np.argsort(np.abs(values))
    values = values[sort_indices]
    states = vectors[:, sort_indices]

    rho_ss_vector = Qobj(states[:, 0])

    rho_ss = vector_to_operator(rho_ss_vector)

    rho_ss.dims = H.dims

    return rho_ss


def transmission_calc(args, results, custom=True):
    fd = args[0]
    params = args[1]
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = collapse_operators(params)
    params.fd = fd
    H = hamiltonian(params)

    if custom:
        if results.shape[0] == 0:
            initial = None
        else:
            idx_min = np.argmin(np.abs(results['fd_points'] - fd))
            initial = results['states'].iloc[idx_min]
        rho_ss = steadystate_custom(H, c_ops, initial)
    else:
        rho_ss = steadystate(H, c_ops)

    rho_c_ss = rho_ss.ptrace(0)
    rho_t_ss = rho_ss.ptrace(1)
    c_occupations = rho_c_ss.diag()
    t_occupations = rho_t_ss.diag()
    edge_occupation_c = c_occupations[params.c_levels - 1]
    edge_occupation_t = t_occupations[params.t_levels - 1]
    transmission = expect(a, rho_ss)
    return np.array([transmission, edge_occupation_c, edge_occupation_t, rho_ss])


def transmission_calc_old(args, results):
    fd = args[0]
    params = args[1]
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = collapse_operators(params)
    params.fd = fd
    H = hamiltonian(params)
    rho_ss = steadystate(H, c_ops)
    rho_c_ss = rho_ss.ptrace(0)
    rho_t_ss = rho_ss.ptrace(1)
    c_occupations = rho_c_ss.diag()
    t_occupations = rho_t_ss.diag()
    edge_occupation_c = c_occupations[params.c_levels - 1]
    edge_occupation_t = t_occupations[params.t_levels - 1]
    transmission = expect(a, rho_ss)
    return np.array([transmission, edge_occupation_c, edge_occupation_t])


def sweep(eps, fd_lower, fd_upper, params, threshold):
    params.eps = eps
    fd_points = np.linspace(fd_lower, fd_upper, 11)
    params_array = np.array([params.copy() for fd in fd_points])
    queue = Queue(params_array, fd_points)
    curvature_iterations = 0
    results = pd.DataFrame()

    while (queue.size > 0) and (curvature_iterations < 3):
        print curvature_iterations
        curvature_iterations = curvature_iterations + 1
        new_results = transmission_calc_array(queue, results)
        results = pd.concat([results, new_results])
        queue.curvature_generate(results, threshold)
    c_levels = [params_instance.c_levels for params_instance in results.params]
    t_levels = [params_instance.t_levels for params_instance in results.params]
    return results


def multi_sweep(eps_array, fd_lower, fd_upper, params, threshold):
    multi_results_dict = dict()

    for eps in eps_array:
        multi_results_dict[eps] = sweep(eps, fd_lower, fd_upper, params, threshold)
        params = multi_results_dict[eps]['params'].iloc[0]
        print params.c_levels
        print params.t_levels

    return multi_results_dict


def qubit_iteration(params, fd_lower=8.9, fd_upper=9.25, display=False):
    threshold = 0.001
    eps = params.eps
    eps_array = np.array([eps])
    multi_results = multi_sweep(eps_array, fd_lower, fd_upper, params, threshold)

    labels = params.labels

    collected_data_re = None
    collected_data_im = None
    collected_data_abs = None
    results_list = []
    for sweep in multi_results.values():
        for i, fd in enumerate(sweep['fd_points'].values):
            transmission = sweep['transmissions'].iloc[i]
            p = sweep['params'].iloc[i]
            coordinates_re = [[fd], [p.eps], [p.Ej], [p.fc], [p.g], [p.kappa], [p.kappa_phi], [p.gamma], [p.gamma_phi],
                              [p.Ec], [p.n_t], [p.n_c]]
            coordinates_im = [[fd], [p.eps], [p.Ej], [p.fc], [p.g], [p.kappa], [p.kappa_phi], [p.gamma], [p.gamma_phi],
                              [p.Ec], [p.n_t], [p.n_c]]
            coordinates_abs = [[fd], [p.eps], [p.Ej], [p.fc], [p.g], [p.kappa], [p.kappa_phi], [p.gamma], [p.gamma_phi],
                               [p.Ec], [p.n_t], [p.n_c]]
            point = np.array([transmission])
            abs_point = np.array([np.abs(transmission)])

            for j in range(len(coordinates_re) - 1):
                point = point[np.newaxis]
                abs_point = abs_point[np.newaxis]

            hilbert_dict = OrderedDict()
            hilbert_dict['t_levels'] = p.t_levels
            hilbert_dict['c_levels'] = p.c_levels
            packaged_point_re = xr.DataArray(point, coords=coordinates_re, dims=labels, attrs=hilbert_dict)
            packaged_point_im = xr.DataArray(point, coords=coordinates_im, dims=labels, attrs=hilbert_dict)
            packaged_point_abs = xr.DataArray(abs_point, coords=coordinates_abs, dims=labels, attrs=hilbert_dict)
            packaged_point_re = packaged_point_re.real
            packaged_point_im = packaged_point_im.imag

            if collected_data_re is not None:
                collected_data_re = collected_data_re.combine_first(packaged_point_re)
            else:
                collected_data_re = packaged_point_re

            if collected_data_im is not None:
                collected_data_im = collected_data_im.combine_first(packaged_point_im)
            else:
                collected_data_im = packaged_point_im

            if collected_data_abs is not None:
                collected_data_abs = collected_data_abs.combine_first(packaged_point_abs)
            else:
                collected_data_abs = packaged_point_abs

    a_abs = collected_data_abs.squeeze()

    if True:

        max_indices = local_maxima(a_abs.values[()])
        maxima = a_abs.values[max_indices]
        indices_order = np.argsort(maxima)

        max_idx = np.argmax(a_abs).values[()]
        A_est = a_abs[max_idx]
        f_r_est = a_abs.f_d[max_idx]
        popt, pcov = lorentzian_fit(a_abs.f_d.values[()], a_abs.values[()])
        f_r = popt[1]

        two_peaks = False
        split = None
        if len(max_indices) >= 2:
            two_peaks = True
            max_indices = max_indices[indices_order[-2:]]

            f_01 = a_abs.f_d[max_indices[1]].values[()]
            f_12 = a_abs.f_d[max_indices[0]].values[()]
            split = f_12 - f_r

    if display:
        fig, axes = plt.subplots(1, 1)
        a_abs.plot(ax=axes)
        plt.show()

        """ 
        fig, axes = plt.subplots(1, 1)
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        xloc = xlim[0] + 0.1*(xlim[1]-xlim[0])
        yloc = ylim[1] - 0.1*(ylim[1]-ylim[0])

        collected_data_abs.plot(ax=axes)
        axes.plot(a_abs.f_d, lorentzian_func(a_abs.f_d, *popt), 'g--')
        print "Resonance frequency = " + str(popt[1]) + " GHz"
        print "Q factor = " + str(Q_factor)
        plt.title(str(p.t_levels) + str(' ') + str(p.c_levels))

        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        if two_peaks == True:
            textstr = '$f_{01}$ = ' + str(f_01) + 'GHz\n' + r'$\alpha$ = ' + str(1000*split) + 'MHz\n$Q$ = ' + str(Q_factor) + '\n$FWHM$ = ' + str(1000*params.kappa) + 'MHz'
        else:
            #textstr = 'fail'
            textstr = '$f_{01}$ = ' + str(f_r_est.values[()]) + 'GHz\n$Q$ = ' + str(
                Q_factor) + '\n$FWHM$ = ' + str(1000 * params.kappa) + 'MHz'

        #textstr = '$f_{01}$ = ' + str(f_01) + 'GHz\n' + r'$\alpha$ = ' + str(split) + 'GHz'
        label = axes.text(xloc, yloc, textstr, fontsize=14, verticalalignment='top', bbox=props)

    plt.show()

    collected_dataset = xr.Dataset({'a_re': collected_data_re,
                                    'a_im': collected_data_im,
                                    'a_abs': collected_data_abs})

    time = datetime.now()
    cwd = os.getcwd()
    time_string = time.strftime('%Y-%m-%d--%H-%M-%S')

    directory = cwd + '/eps=' + str(eps) + 'GHz' + '/' + time_string
    if not os.path.exists(directory):
        os.makedirs(directory)
        collected_dataset.to_netcdf(directory+'/spectrum.nc')

    """

    # new_fq = params.fq + 9.19324 - f_r_est.values[()]
    # new_chi = (2*params.chi - split - 0.20356)/2
    # new_chi = -0.20356 * params.chi / split

    return f_r, split


def cavity_iteration(params, fd_lower=10.47, fd_upper=10.51, display=False):
    threshold = 0.0005

    eps = params.eps
    eps_array = np.array([eps])

    multi_results = multi_sweep(eps_array, fd_lower, fd_upper, params, threshold)

    labels = params.labels

    collected_data_re = None
    collected_data_im = None
    collected_data_abs = None
    results_list = []
    for sweep in multi_results.values():
        for i, fd in enumerate(sweep['fd_points'].values):
            transmission = sweep['transmissions'].iloc[i]
            p = sweep['params'].iloc[i]
            coordinates_re = [[fd], [p.eps], [p.Ej], [p.fc], [p.g], [p.kappa], [p.kappa_phi], [p.gamma], [p.gamma_phi],
                              [p.Ec], [p.n_t], [p.n_c]]
            coordinates_im = [[fd], [p.eps], [p.Ej], [p.fc], [p.g], [p.kappa], [p.kappa_phi], [p.gamma], [p.gamma_phi],
                              [p.Ec], [p.n_t], [p.n_c]]
            coordinates_abs = [[fd], [p.eps], [p.Ej], [p.fc], [p.g], [p.kappa], [p.kappa_phi], [p.gamma], [p.gamma_phi],
                               [p.Ec], [p.n_t], [p.n_c]]
            point = np.array([transmission])
            abs_point = np.array([np.abs(transmission)])

            for j in range(len(coordinates_re) - 1):
                point = point[np.newaxis]
                abs_point = abs_point[np.newaxis]

            hilbert_dict = OrderedDict()
            hilbert_dict['t_levels'] = p.t_levels
            hilbert_dict['c_levels'] = p.c_levels
            packaged_point_re = xr.DataArray(point, coords=coordinates_re, dims=labels, attrs=hilbert_dict)
            packaged_point_im = xr.DataArray(point, coords=coordinates_im, dims=labels, attrs=hilbert_dict)
            packaged_point_abs = xr.DataArray(abs_point, coords=coordinates_abs, dims=labels, attrs=hilbert_dict)
            packaged_point_re = packaged_point_re.real
            packaged_point_im = packaged_point_im.imag

            if collected_data_re is not None:
                collected_data_re = collected_data_re.combine_first(packaged_point_re)
            else:
                collected_data_re = packaged_point_re

            if collected_data_im is not None:
                collected_data_im = collected_data_im.combine_first(packaged_point_im)
            else:
                collected_data_im = packaged_point_im

            if collected_data_abs is not None:
                collected_data_abs = collected_data_abs.combine_first(packaged_point_abs)
            else:
                collected_data_abs = packaged_point_abs

    a_abs = collected_data_abs.squeeze()

    max_indices = local_maxima(a_abs.values[()])
    maxima = a_abs.values[max_indices]
    indices_order = np.argsort(maxima)

    two_peaks = False
    if len(max_indices) == 2:
        two_peaks = True

        max_indices = max_indices[indices_order[-2:]]

        f_r = a_abs.f_d[max_indices[1]].values[()]
        f_r_2 = a_abs.f_d[max_indices[0]].values[()]
        split = f_r - f_r_2

        ratio = a_abs[max_indices[1]] / a_abs[max_indices[0]]
        ratio = ratio.values[()]

    max_idx = np.argmax(a_abs).values[()]
    A_est = a_abs[max_idx]
    f_r_est = a_abs.f_d[max_idx]
    # popt, pcov = curve_fit(lorentzian_func, a_abs.f_d, a_abs.values, p0=[A_est, f_r_est, 0.001])
    popt, pcov = lorentzian_fit(a_abs.f_d.values[()], a_abs.values[()])
    Q_factor = popt[2]

    if display:
        fig, axes = plt.subplots(1, 1)
        a_abs.plot(ax=axes)
        plt.show()

    """
    print "Resonance frequency = " + str(popt[1]) + " GHz"
    print "Q factor = " + str(Q_factor)

    fig, axes = plt.subplots(1,1)
    collected_data_abs.plot(ax=axes)
    axes.plot(a_abs.f_d, lorentzian_func(a_abs.f_d, *popt), 'g--')

    plt.title(str(p.t_levels) + str(' ') + str(p.c_levels))

    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    if two_peaks == True:
        textstr = 'f_r = ' + str(popt[1]) + 'GHz\n$Q$ = ' + str(Q_factor) + '\n$\chi$ = ' + str(
            split * 1000) + 'MHz\n$\kappa$ = ' + str(1000 * params.kappa) + 'MHz\nRatio = ' + str(ratio)
    else:
        textstr = 'f_r = ' + str(popt[1]) + 'GHz\n$Q$ = ' + str(Q_factor) + '\n$\kappa$ = ' + str(1000 * params.kappa) + 'MHz'

    label = axes.text(a_abs.f_d[0], popt[0], textstr, fontsize=14, verticalalignment='top', bbox=props)

    #collected_dataset = xr.Dataset({'a_re': collected_data_re,
    #                                'a_im': collected_data_im,
    #                                'a_abs': collected_data_abs})

    #time = datetime.now()
    #cwd = os.getcwd()
    #time_string = time.strftime('%Y-%m-%d--%H-%M-%S')

    #directory = cwd + '/eps=' + str(eps) + 'GHz' + '/' + time_string
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    #    collected_dataset.to_netcdf(directory+'/spectrum.nc')

    plt.show()

    """

    # fc_new = params.fc + 10.49602 - popt[1]
    # g_new = params.g * np.sqrt(23.8 * 1000 / split) / 1000
    # kappa_new = Q_factor * params.kappa / 8700

    return popt[1], split, Q_factor