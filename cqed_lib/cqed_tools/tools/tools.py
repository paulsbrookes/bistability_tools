from qutip import *
import scipy
import pandas as pd
import numpy as np
from copy import deepcopy
from ..simulation import hamiltonian, maximum_finder


def add_column(df, op, name):
    new_column = []
    for state in df[0]:
        new_column.append(expect(state,op))
    df[name] = new_column
    return df


def x1_rate_calc(x, eliminated_params):
    x1, x2 = x
    x1_rate = -eliminated_params['kappa']*x1 + eliminated_params['delta']*x2 + 0.5*eliminated_params['chi']*(x1**2 + x2**2)*x2 + 2*eliminated_params['eps'].imag
    return x1_rate


def x2_rate_calc(x, eliminated_params):
    x1, x2 = x
    x2_rate = -eliminated_params['delta']*x1 - eliminated_params['kappa']*x2 - 0.5*eliminated_params['chi']*(x1**2 + x2**2)*x1 - 2*eliminated_params['eps'].real
    return x2_rate


def classical_eom(x, eliminated_params):
    x1_rate = x1_rate_calc(x, eliminated_params)
    x2_rate = x2_rate_calc(x, eliminated_params)
    return (x1_rate, x2_rate)


def package_state(state_array, dims):
    packaged_state = Qobj(state_array)
    packaged_state.dims = dims
    packaged_state = vector_to_operator(packaged_state)
    return packaged_state


def ham_gen_eliminated(params):
    b = destroy(int(params['t_levels']))
    ham = params['delta']*b.dag()*b + params['chi']*b.dag()*b.dag()*b*b + params['eps']*b + np.conjugate(params['eps'])*b.dag()
    return ham


def c_ops_gen_eliminated(params):
    b = destroy(int(params['t_levels']))
    c_ops = []
    c_ops.append(np.sqrt(2*params['kappa'])*b)
    return c_ops


def ham_gen_full(params):
    a = tensor(qeye(int(params['t_levels'])), destroy(int(params['c_levels'])))
    b = tensor(destroy(int(params['t_levels'])), qeye(int(params['c_levels'])))
    ham = params['delta_a']*a.dag()*a + params['delta_b']*b.dag()*b + params['chi']*b.dag()*b.dag()*b*b + params['g']*(a*b.dag() + a.dag()*b) + params['eps']*(a + a.dag())
    return ham


def c_ops_gen_full(params):
    a = tensor(qeye(int(params['t_levels'])), destroy(int(params['c_levels'])))
    b = tensor(destroy(int(params['t_levels'])), qeye(int(params['c_levels'])))
    c_ops = []
    c_ops.append(np.sqrt(2*params['gamma'])*a)
    c_ops.append(np.sqrt(2*params['kappa'])*b)
    return c_ops


def eliminate(params):
    delta_a = params['delta_a']
    delta_b = params['delta_b']
    delta_eff = delta_b - params['g']**2 * delta_a/(delta_a**2 + params['kappa']**2)
    kappa_eff = params['gamma'] + params['g']**2 * params['kappa']/(delta_a**2 + params['kappa']**2)
    eps_1_eff = params['g']*params['kappa']*params['eps']/(delta_a**2 + params['kappa']**2)
    eps_2_eff = params['g']*delta_a*params['eps']/(delta_a**2 + params['kappa']**2)
    eps_eff = -(1j*eps_1_eff + eps_2_eff)
    params_list = [delta_eff, params['chi'], eps_eff, kappa_eff, params['t_levels']]
    names_list = ['delta', 'chi', 'eps', 'kappa', 't_levels']
    eliminated_params = pd.Series(params_list, names_list)
    return eliminated_params


def lowest_occupation_calc(C, steady_state, adr_state):
    rho = steady_state + C*adr_state
    occupations = rho.eigenenergies(eigvals=1)
    lowest_occupation = np.min(occupations)
    return lowest_occupation


def curve_calc_old(C, steady_state, adr_state, bounds=None):
    delta = 0.001
    C_array = C + delta*np.arange(-1,2)
    lowest_occupations = []
    for C in C_array:
        rho = steady_state + C*adr_state
        occupations, states = rho.eigenstates()
        lowest_occupations.append(np.min(occupations))
    curvature = np.diff(lowest_occupations,2)
    if bounds is not None:
        if C > bounds[1]:
            curvature += (C-bounds[1])*10
        elif C < bounds[0]:
            curvature += (C-bounds[0])*10
    return curvature


def curve_calc(C, steady_state, adr_state, region=None):
    if region is 'upper' and C < 0:
        return -C
    elif region is 'lower' and C > 0:
        return C
    else:
        delta = 0.001
        epsilon = 0.1
        C_array = C + delta*np.arange(0,2)
        lowest_occupations = []
        for C in C_array:
            rho = steady_state + C*adr_state
            occupations = rho.eigenenergies(eigvals=1)
            lowest_occupations.append(np.min(occupations))
        curvature = np.diff(lowest_occupations,1)
        objective = -lowest_occupations[0]
        if np.abs(curvature[0]) < 1e-5:
            objective -= np.abs(C)
        return objective


def metastable_calc(steady_state, adr_state, return_coefficients=False):
    adr_state /= np.sqrt((adr_state**2).tr())
    res1 = scipy.optimize.minimize(curve_calc, 2.0, method='Nelder-Mead', args=(steady_state, adr_state, 'lower'))
    res2 = scipy.optimize.minimize(curve_calc, -2.0, method='Nelder-Mead', args=(steady_state, adr_state, 'upper'))
    if np.isclose(res1.x[0], res2.x[0]):
        print('Only one metastable state has been identified.')
        return None, None
    metastable1 = steady_state + res1.x[0]*adr_state
    metastable2 = steady_state + res2.x[0]*adr_state
    meta_states = np.array([metastable1, metastable2])
    a = destroy(steady_state.dims[0][0])
    amplitudes = [np.abs(expect(state, a)) for state in meta_states]
    sort_indices = np.argsort(amplitudes)
    meta_states = meta_states[sort_indices]
    if return_coefficients:
        return meta_states[0], meta_states[1], [res1.x[0], res2.x[0]]
    else:
        return meta_states[0], meta_states[1]


def rates_calc(L, rho_d, rho_b):
    adr_vector = operator_to_vector(rho_d-rho_b)
    norm = adr_vector.norm()
    bright_vector = operator_to_vector(rho_b)
    dim_vector = operator_to_vector(rho_d)
    adr_vector.dims = L.dims
    bright_vector.dims = L.dims
    rate_bd = adr_vector.dag()*L*bright_vector / norm**2
    rate_bd = np.abs(rate_bd[0,0])
    rate_db = adr_vector.dag()*L*dim_vector / norm**2
    rate_db = np.abs(rate_db[0,0])
    return rate_db, rate_bd


def lowest_occupation_calc(C, steady_state, adr_state):
    rho = steady_state + C*adr_state
    occupations = rho.eigenenergies(eigvals=1)
    lowest_occupation = np.min(occupations)
    return lowest_occupation


def diagonalise_subspace(op, basis):
    n_states = basis.shape[0]
    matrix_elements = np.zeros([n_states, n_states], dtype=np.complex128)
    for i in range(n_states):
        for j in range(n_states):
            matrix_elements[i, j] = (basis[i].dag()*op*basis[j])[0,0]
    op_subspace = Qobj(matrix_elements)
    eigenvalues, eigenvectors = op_subspace.eigenstates()
    new_basis = []
    for vector in eigenvectors:
        new_basis += [np.sum(vector.full()[:, 0] * basis)]
    new_basis = np.array(new_basis)
    return eigenvalues, new_basis


def calculate_constants(directory, hermitianize=False):
    c_ops = qload(directory+'/c_ops')
    H = qload(directory+'/slowdown_hamiltonian')
    L = liouvillian(H,c_ops)
    rho_ss = qload(directory+'/steady_state')
    rho_final = qload(directory+'/state_checkpoint')
    eigenvalues, new_basis = calc_adr_state(rho_ss, rho_final, L)
    adr_idx = np.argmax(np.abs(eigenvalues))
    rate = eigenvalues[adr_idx]
    return -1/(2*np.pi*1000*rate.real)


def calc_adr_state(rho_ss, rho_final, L, hermitianize=False):
    rho_ss = operator_to_vector(rho_ss)
    rho_ss /= rho_ss.norm()
    if not hermitianize:
        rho_final = operator_to_vector(rho_final)
    else:
        rho_final = 0.5*(rho_final + rho_final.dag())
        rho_final = operator_to_vector(rho_final)
    rho_final /= rho_final.norm()
    overlap = (rho_ss.dag()*rho_final)[0,0]
    rho_ortho = rho_final - overlap*rho_ss
    rho_ortho /= rho_ortho.norm()
    basis = np.array([rho_ss, rho_ortho])
    eigenvalues, new_basis = diagonalise_subspace(L, basis)
    return eigenvalues, new_basis


def expand(rho, new_c_levels):
    old_c_levels = rho.dims[0][0]
    t_levels = rho.dims[0][1]
    n_zeros = t_levels * (new_c_levels - old_c_levels)
    padded_contents = np.pad(rho.full(), ((0,n_zeros),(0,n_zeros)), 'constant')
    new_state = Qobj(padded_contents)
    new_state.dims = [[new_c_levels, t_levels], [new_c_levels, t_levels]]
    return new_state


def isolate_metastable(params, alpha, truncated_c_levels, return_alpha=None):
    params_truncated = deepcopy(params)
    params_truncated.c_levels = truncated_c_levels
    ham = hamiltonian(params_truncated, alpha=alpha)
    c_ops = collapse_operators(params_truncated, alpha=alpha)
    rho_ss_truncated = steadystate(ham, c_ops)
    rho_ss = expand(rho_ss_truncated, params.c_levels)
    if return_alpha is None:
        displacement_op = tensor(displace(params.c_levels, alpha), qeye(params.t_levels))
    else:
        displacement_op = tensor(displace(params.c_levels, return_alpha), qeye(params.t_levels))
    rho_ss = displacement_op*rho_ss*displacement_op.dag()
    return rho_ss


def metastable_calc_optimization(rho_ss, rho_adr):
    rho_ss /= rho_ss.tr()
    rho_adr /= rho_adr.tr()

    rho_c_ss = rho_ss.ptrace(0)
    rho_c_adr = rho_adr.ptrace(0)

    res = scipy.optimize.minimize(objective_calc, 0.0, method='Nelder-Mead', args=(rho_c_ss, rho_c_adr))
    rho_d = rho_ss + res.x[0] * rho_adr
    rho_d /= rho_d.tr()

    rho_2 = rho_d
    rho_c_2 = rho_2.ptrace(0)

    rho_1 = rho_adr
    rho_1 -= rho_2 * (rho_1 * rho_2).tr() / (rho_2 * rho_2).tr()
    rho_c_1 = rho_1.ptrace(0)
    res = scipy.optimize.minimize(objective_calc, 0.0, method='Nelder-Mead', args=(rho_c_1, rho_c_2))
    rho_b_adr = rho_1 + res.x[0] * rho_2
    rho_b_adr /= rho_b_adr.tr()

    rho_1 = rho_ss
    rho_1 -= rho_2 * (rho_1 * rho_2).tr() / (rho_2 * rho_2).tr()
    rho_c_1 = rho_1.ptrace(0)
    res = scipy.optimize.minimize(objective_calc, 0.0, method='Nelder-Mead', args=(rho_c_1, rho_c_2))
    rho_b_ss = rho_1 + res.x[0] * rho_2
    rho_b_ss /= rho_b_ss.tr()

    states_b = [rho_b_ss, rho_b_adr]
    distances = [tracedist(rho_b, rho_d) for rho_b in states_b]
    rho_b = states_b[np.argmax(distances)]

    c_levels = rho_b.dims[0][0]
    t_levels = rho_d.dims[0][1]
    a = tensor(destroy(c_levels), qeye(t_levels))
    a_exp = [np.abs(expect(a, rho_d)), np.abs(expect(a, rho_b))]
    states = np.array([rho_d, rho_b])
    states = states[np.argsort(a_exp)]

    return states[0], states[1]


def ratio_calc(state, n_bins=100, return_peaks=False):
    xvec = np.linspace(-8, 8, n_bins)
    W = np.abs(wigner(state, xvec, xvec, g=2))
    W[40:60, 45:70] = 0
    peak_indices = maximum_finder(W)
    peak_heights = np.array([W[peak_indices[0][idx], peak_indices[1][idx]] for idx in range(len(peak_indices[0]))])
    peaks = pd.DataFrame(np.array([peak_indices[0], peak_indices[1], peak_heights]).T, columns=['i', 'j', 'height'])
    dtypes = {'i': int, 'j': int, 'height': float}
    peaks = peaks.astype(dtypes)
    peaks.sort_values(by='height', axis=0, ascending=False, inplace=True)

    i_diff = peaks.iloc[0].i - peaks.iloc[1].i
    j_diff = peaks.iloc[0].j - peaks.iloc[1].j
    dist = np.sqrt(i_diff ** 2 + j_diff ** 2)
    if dist < 10:
        index = peaks.index[1]
        peaks.drop(index=index, inplace=True)

    ratio = peaks['height'].iloc[1] / peaks['height'].iloc[0]
    if return_peaks:
        return ratio, peaks
    else:
        return ratio


def objective_calc(x, rho1, rho2):
    rho = rho1 + x[0] * rho2
    ratio = ratio_calc(rho)
    return ratio


def prob_objective_calc(x, rho_d, rho_b, rho_steady):
    rho = x[0] * rho_d + (1 - x[0]) * rho_b
    distance = tracedist(rho, rho_steady)
    return distance