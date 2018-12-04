import numpy as np
from qutip import *
from scipy.optimize import fsolve, root, minimize
from ..simulation.hamiltonian_gen import coupling_calc
from cqed_tools.mf.hamiltonian_gen_mf import hamiltonian_mf, collapse_operators_mf


def dalpha_calc_me(x, L=None, ham=None, c_ops=None):
    alpha, beta = x
    if L is not None:
        c_levels = L.dims[0][0][0]
        t_levels = L.dims[0][0][1]
        rho = tensor(coherent_dm(c_levels, alpha), coherent_dm(t_levels, beta))
        drho = vector_to_operator(L * operator_to_vector(rho))
    else:
        c_levels = ham.dims[0][0]
        t_levels = ham.dims[0][1]
        rho = tensor(coherent_dm(c_levels, alpha), coherent_dm(t_levels, beta))
        drho = lindblad_me(rho, ham, c_ops)
    a = tensor(destroy(c_levels), qeye(t_levels))
    b = tensor(qeye(c_levels), destroy(t_levels))
    dalpha = (a * drho).tr()
    dbeta = (b * drho).tr()
    return dalpha, dbeta


def classical_eom_me(x, L, magnitude=False):
    alpha = x[0] + 1j * x[1]
    beta = x[2] + 1j * x[3]
    dalpha, dbeta = dalpha_calc_me([alpha, beta], L=L)
    dx = np.array([dalpha.real, dalpha.imag, dbeta.real, dbeta.imag])
    if magnitude:
        return np.linalg.norm(dx)
    else:
        return dx


def locate_fixed_point(L, alpha0=(0, 0), beta0=(0, 0)):
    x0 = np.array([alpha0[0], alpha0[1], beta0[0], beta0[1]])
    res = root(classical_eom_me, x0, args=(L,), method='hybr')
    alpha = res.x[0] + 1j * res.x[1]
    beta = res.x[2] + 1j * res.x[3]
    return alpha, beta


def dalpha_calc_mf(params, alpha, beta):
    rho = coherent_dm(params.t_levels, beta)
    lower_levels = np.arange(0, params.t_levels - 1)
    upper_levels = np.arange(1, params.t_levels)
    q = -params.Ej / (2 * params.Ec)
    coupling_array = coupling_calc(lower_levels, upper_levels, q)
    coupling_array = coupling_array / coupling_array[0]
    down_transmon_transitions = 0
    for i, coupling in enumerate(coupling_array):
        down_transmon_transitions += coupling * basis(params.t_levels, i) * basis(params.t_levels, i + 1).dag()
    down_transmon_transitions *= params.g
    dalpha = -1j * (
    (params.fc - params.fd) * alpha + params.eps + expect(down_transmon_transitions, rho)) - 0.5 * params.kappa * alpha
    return dalpha


def dbeta_calc_mf(params, alpha, beta):
    rho = coherent_dm(params.t_levels, beta)
    ham_mf = hamiltonian_mf(params, alpha)
    b = destroy(params.t_levels)
    c_ops_mf = collapse_operators_mf(params)
    L = liouvillian(ham_mf, c_ops_mf)
    drho = vector_to_operator(L * operator_to_vector(rho))
    dbeta = (b * drho).tr()
    return dbeta


def classical_eom_mf(x, params):
    alpha = x[0] + 1j * x[1]
    beta = x[2] + 1j * x[3]
    dalpha = dalpha_calc_mf(params, alpha, beta)
    dbeta = dbeta_calc_mf(params, alpha, beta)
    dx = np.array([dalpha.real, dalpha.imag, dbeta.real, dbeta.imag])
    return dx


def locate_fixed_point_mf(params, alpha0=(0, 0), beta0=(0, 0)):
    x0 = np.array([alpha0[0], alpha0[1], beta0[0], beta0[1]])
    res = root(classical_eom_mf, x0, args=(params,), method='hybr')
    alpha = res.x[0] + 1j * res.x[1]
    beta = res.x[2] + 1j * res.x[3]
    return alpha, beta


def fixed_point_tracer(fd_array, params, alpha0=0, beta0=0):
    amplitude_array = np.zeros([fd_array.shape[0], 2], dtype=complex)
    for idx, fd in enumerate(fd_array):
        print(idx)
        params_instance = deepcopy(params)
        params_instance.fd = fd
        alpha_fixed, beta_fixed = locate_fixed_point_mf(params_instance, alpha0=[alpha0.real, alpha0.imag], beta0=[beta0.real, beta0.imag])
        amplitude_array[idx, :] = [alpha_fixed, beta_fixed]
        alpha0, beta0 = alpha_fixed, beta_fixed
    amplitude_frame = pd.DataFrame(amplitude_array, index=fd_array, columns=['a', 'b'])
    amplitude_frame.sort_index(inplace=True)
    return amplitude_frame