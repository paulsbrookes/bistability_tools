import numpy as np
from qutip import *
from scipy.optimize import root
from scipy.special import factorial
from scipy import sparse, interpolate
from ..simulation.hamiltonian_gen import coupling_calc
from cqed_tools.mf.hamiltonian_gen_mf import hamiltonian_mf, collapse_operators_mf
from copy import deepcopy
import pandas as pd
from tqdm import tqdm



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
    if res.success:
        alpha = res.x[0] + 1j * res.x[1]
        beta = res.x[2] + 1j * res.x[3]
    else:
        alpha, beta = None, None
    return alpha, beta


def fixed_point_tracker(fd_array, params, alpha0=0, beta0=0, consistency_check=True, fill_value=None, threshold=1e-4,
                        columns=['a', 'b'], crosscheck_frame=None):
    amplitude_array = np.zeros([fd_array.shape[0], 2], dtype=complex)
    trip = False
    for idx, fd in tqdm(enumerate(fd_array)):
        if not trip:
            params_instance = deepcopy(params)
            params_instance.fd = fd
            alpha_fixed, beta_fixed = locate_fixed_point_mf(params_instance, alpha0=[alpha0.real, alpha0.imag],
                                                            beta0=[beta0.real, beta0.imag])
            if alpha_fixed is None:
                trip = True
                amplitude_array[idx, :] = [fill_value, fill_value]

            if not trip:
                if consistency_check:
                    params_check = deepcopy(params_instance)
                    params_check.t_levels += 1
                    alpha_check, beta_check = locate_fixed_point_mf(params_instance,
                                                                    alpha0=[alpha_fixed.real, alpha_fixed.imag],
                                                                    beta0=[beta_fixed.real, beta_fixed.imag])

                    converged = np.abs(alpha_check - alpha_fixed) < threshold and np.abs(
                        alpha_check - alpha_fixed) < threshold
                    if not converged:
                        trip = True

                    if crosscheck_frame is not None:
                        alpha_crosscheck = crosscheck_frame['a'].iloc[idx]
                        beta_crosscheck = crosscheck_frame['b'].iloc[idx]
                        already_found = np.abs(alpha_crosscheck - alpha_fixed) < threshold and np.abs(
                            alpha_crosscheck - alpha_fixed) < threshold
                        if already_found:
                            trip = True

                if trip:
                    amplitude_array[idx, :] = [fill_value, fill_value]
                else:
                    amplitude_array[idx, :] = [alpha_fixed, beta_fixed]
                    alpha0, beta0 = alpha_fixed, beta_fixed

        else:
            amplitude_array[idx, :] = [fill_value, fill_value]
    amplitude_frame = pd.DataFrame(amplitude_array, index=fd_array, columns=columns)
    amplitude_frame.sort_index(inplace=True)
    return amplitude_frame


def mf_characterise(base_params, fd_array):
    alpha0 = 0
    beta0 = 0
    mf_amplitude_frame_bright = fixed_point_tracker(np.flip(fd_array, axis=0), base_params, alpha0=alpha0, beta0=beta0)
    mf_amplitude_frame_dim = fixed_point_tracker(fd_array, base_params, alpha0=alpha0, beta0=beta0,
                                                 columns=['a_dim', 'b_dim'], crosscheck_frame=mf_amplitude_frame_bright)
    mf_amplitude_frame_bright.columns = ['a_bright', 'b_bright']
    mf_amplitude_frame = pd.concat([mf_amplitude_frame_bright, mf_amplitude_frame_dim], axis=1)
    return mf_amplitude_frame


def mf_characterise(base_params, fd_array, alpha0_bright=0, beta0_bright=0, alpha0_dim=0, beta0_dim=0):
    mf_amplitude_frame_bright = fixed_point_tracker(np.flip(fd_array, axis=0), base_params, alpha0=alpha0_bright,
                                                    beta0=beta0_bright)
    mf_amplitude_frame_dim = fixed_point_tracker(fd_array, base_params, alpha0=alpha0_dim, beta0=beta0_dim,
                                                 columns=['a_dim', 'b_dim'], crosscheck_frame=mf_amplitude_frame_bright)
    mf_amplitude_frame_bright.columns = ['a_bright', 'b_bright']
    mf_amplitude_frame = pd.concat([mf_amplitude_frame_bright, mf_amplitude_frame_dim], axis=1)
    return mf_amplitude_frame