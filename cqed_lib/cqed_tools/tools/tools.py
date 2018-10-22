from qutip import *
import numpy as np
import scipy
import pandas as pd
import numpy as np


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


def metastable_calc(steady_state, adr_state):
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