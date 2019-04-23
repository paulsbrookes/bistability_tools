import numpy as np
from qutip import *
from scipy.optimize import root
from scipy.special import factorial
from scipy import sparse, interpolate
from ..simulation.hamiltonian_gen import coupling_calc
from cqed_tools.mf.hamiltonian_gen_mf import hamiltonian_mf, collapse_operators_mf, c_matrices_gen
from copy import deepcopy
import pandas as pd
from tqdm import tqdm


def dalpha_calc_mf(alpha, beta, params, c_down):
    dalpha = -1j * (
        (params.fc - params.fd) * alpha + params.eps + compute_poly(beta, c_down)) - 0.5 * params.kappa * alpha
    return dalpha


def dbeta_calc_mf(alpha, beta, c_0, c_alpha, c_alpha_conj, params):
    poly_0 = compute_poly(beta, c_0)
    poly_alpha = compute_poly(beta, c_alpha)
    poly_alpha_conj = compute_poly(beta, c_alpha_conj)
    return poly_0 + alpha * poly_alpha + np.conjugate(alpha) * poly_alpha_conj + 1j * params.fd * beta


def locate_fixed_point_mf(params, c_matrices, alpha0=(0, 0), beta0=(0, 0)):
    c_0 = c_matrices['c_0']
    c_alpha = c_matrices['c_alpha']
    c_alpha_conj = c_matrices['c_alpha_conj']
    c_down = c_matrices['c_down']
    x0 = np.array([alpha0[0], alpha0[1], beta0[0], beta0[1]])
    res = root(classical_eom_mf, x0, args=(params, c_0, c_alpha, c_alpha_conj, c_down), method='hybr')
    if res.success:
        alpha = res.x[0] + 1j * res.x[1]
        beta = res.x[2] + 1j * res.x[3]
    else:
        alpha, beta = None, None
    return alpha, beta


def classical_eom_mf(x, params, c_0, c_alpha, c_alpha_conj, c_down):
    alpha = x[0] + 1j * x[1]
    beta = x[2] + 1j * x[3]
    dalpha = dalpha_calc_mf(alpha, beta, params, c_down)
    dbeta = dbeta_calc_mf(alpha, beta, c_0, c_alpha, c_alpha_conj, params)
    dx = np.array([dalpha.real, dalpha.imag, dbeta.real, dbeta.imag])
    return dx


def fixed_point_tracker(fd_array, params, c_matrices, alpha0=0, beta0=0, fill_value=None, threshold=1e-4,
                        columns=['a', 'b']):
    amplitude_array = np.zeros([fd_array.shape[0], 2], dtype=complex)
    trip = False
    found_first_point = False
    for idx, fd in tqdm(enumerate(fd_array)):
        if not trip:
            params_instance = deepcopy(params)
            params_instance.fd = fd
            alpha_fixed, beta_fixed = locate_fixed_point_mf(params_instance, c_matrices,
                                                            alpha0=[alpha0.real, alpha0.imag],
                                                            beta0=[beta0.real, beta0.imag])
            if alpha_fixed is None:
                if found_first_point == True:
                    trip = True
                amplitude_array[idx, :] = [fill_value, fill_value]
            else:
                found_first_point = True
                # print(alpha_fixed)
                amplitude_array[idx, :] = [alpha_fixed, beta_fixed]
                alpha0 = alpha_fixed
                beta0 = beta_fixed
        else:
            amplitude_array[idx, :] = [fill_value, fill_value]
    amplitude_frame = pd.DataFrame(amplitude_array, index=fd_array, columns=columns)
    amplitude_frame.sort_index(inplace=True)
    return amplitude_frame


def mf_characterise(base_params, fd_array, alpha0_bright=0, beta0_bright=0, alpha0_dim=0, beta0_dim=0, c_matrices=None, duffing=False):
    if c_matrices is None:
        c_matrices = c_matrices_gen(base_params, duffing=duffing)
    mf_amplitude_frame_bright = fixed_point_tracker(np.flip(fd_array, axis=0), base_params, c_matrices, alpha0=alpha0_bright, beta0=beta0_bright)
    mf_amplitude_frame_dim = fixed_point_tracker(fd_array, base_params, c_matrices, alpha0=alpha0_dim, beta0=beta0_dim, columns=['a_dim', 'b_dim'])
    mf_amplitude_frame_bright.columns = ['a_bright', 'b_bright']
    mf_amplitude_frame = pd.concat([mf_amplitude_frame_bright, mf_amplitude_frame_dim], axis=1)

    overlap_find = ~np.isclose(mf_amplitude_frame['a_bright'].values, mf_amplitude_frame['a_dim'].values)

    if not np.any(overlap_find):
        mf_amplitude_frame = pd.DataFrame(mf_amplitude_frame.values[:, 0], index=mf_amplitude_frame.index,
                                          columns=['a'])
    else:
        start_idx = np.where(overlap_find)[0][0]
        end_idx = np.where(overlap_find)[0][-1]

        mf_amplitude_frame['a_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['b_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['a_dim'].iloc[end_idx + 1:] = None
        mf_amplitude_frame['b_dim'].iloc[end_idx + 1:] = None

    return mf_amplitude_frame


























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


def find_overlap(mf_amplitude_frame, params):
    alpha0_dim = mf_amplitude_frame['a_dim'].dropna().iloc[-1]
    beta0_dim = mf_amplitude_frame['b_dim'].dropna().iloc[-1]
    fd_lower = mf_amplitude_frame['a_dim'].dropna().index[-1]
    alpha0_bright = mf_amplitude_frame['a_bright'].dropna().iloc[0]
    beta0_bright = mf_amplitude_frame['b_bright'].dropna().iloc[0]
    fd_upper = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_array = np.linspace(fd_lower, fd_upper, 5)[1:-1]
    new_mf_amplitude_frame = mf_characterise(params, fd_array, alpha0_bright=alpha0_bright, beta0_bright=beta0_bright,
                                             alpha0_dim=alpha0_dim, beta0_dim=beta0_dim)
    combined = pd.concat([mf_amplitude_frame, new_mf_amplitude_frame])
    combined.sort_index(inplace=True)
    return combined


def extend_lower(mf_amplitude_frame, params):
    indices = mf_amplitude_frame.index
    fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
    fd_lower1 = indices[fd_lower1_idx]
    fd_lower = 0.5 * (fd_lower1 + fd_lower2)
    alpha0_bright = mf_amplitude_frame['a_bright'].dropna().iloc[0]
    beta0_bright = mf_amplitude_frame['b_bright'].dropna().iloc[0]
    alpha0_dim = 0.5 * (
    mf_amplitude_frame['a_dim'].iloc[fd_lower1_idx] + mf_amplitude_frame['a_dim'].iloc[fd_lower1_idx + 1])
    beta0_dim = 0.5 * (
    mf_amplitude_frame['b_dim'].iloc[fd_lower1_idx] + mf_amplitude_frame['b_dim'].iloc[fd_lower1_idx + 1])
    fd_array = np.array([fd_lower])
    new_mf_amplitude_frame = mf_characterise(params, fd_array, alpha0_bright=alpha0_bright, beta0_bright=beta0_bright,
                                             alpha0_dim=alpha0_dim, beta0_dim=beta0_dim)
    combined = pd.concat([mf_amplitude_frame, new_mf_amplitude_frame])
    combined.sort_index(inplace=True)
    return combined


def extend_upper(mf_amplitude_frame, params):
    indices = mf_amplitude_frame.index
    fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
    fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
    fd_upper2 = indices[fd_upper2_idx]
    fd_upper = 0.5 * (fd_upper1 + fd_upper2)
    alpha0_dim = mf_amplitude_frame['a_dim'].dropna().iloc[-1]
    beta0_dim = mf_amplitude_frame['b_dim'].dropna().iloc[-1]
    alpha0_bright = 0.5 * (
    mf_amplitude_frame['a_bright'].iloc[fd_upper2_idx] + mf_amplitude_frame['a_bright'].iloc[fd_upper2_idx - 1])
    beta0_bright = 0.5 * (
    mf_amplitude_frame['b_bright'].iloc[fd_upper2_idx] + mf_amplitude_frame['b_bright'].iloc[fd_upper2_idx - 1])
    fd_array = np.array([fd_upper])
    new_mf_amplitude_frame = mf_characterise(params, fd_array, alpha0_bright=alpha0_bright, beta0_bright=beta0_bright,
                                             alpha0_dim=alpha0_dim, beta0_dim=beta0_dim)
    combined = pd.concat([mf_amplitude_frame, new_mf_amplitude_frame])
    combined.sort_index(inplace=True)
    return combined


def check_upper(mf_amplitude_frame, params, c_matrices):
    indices = mf_amplitude_frame.index
    fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
    fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
    fd_upper2 = indices[fd_upper2_idx]
    alpha0_dim = mf_amplitude_frame['a_dim'].iloc[fd_upper2_idx - 1]
    beta0_dim = mf_amplitude_frame['b_dim'].iloc[fd_upper2_idx - 1]
    fd_array = np.array([fd_upper2])
    mf_amplitude_frame_dim = fixed_point_tracker(np.flip(fd_array, axis=0), params, c_matrices, alpha0=alpha0_dim, beta0=beta0_dim)
    if mf_amplitude_frame_dim.dropna().shape[0] and not np.all(np.isclose(mf_amplitude_frame_dim.iloc[0].values, mf_amplitude_frame.iloc[fd_upper2_idx][['a_bright', 'b_bright']].values)):
        mf_amplitude_frame.loc[fd_upper2][['a_dim', 'b_dim']] = mf_amplitude_frame_dim.values[0, :]
        success = True
    else:
        success = False
    return mf_amplitude_frame, success


def check_lower(mf_amplitude_frame, params, c_matrices):
    indices = mf_amplitude_frame.index
    fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
    fd_lower1 = indices[fd_lower1_idx]
    alpha0_bright = mf_amplitude_frame['a_bright'].iloc[fd_lower1_idx + 1]
    beta0_bright = mf_amplitude_frame['b_bright'].iloc[fd_lower1_idx + 1]
    fd_array = np.array([fd_lower1])
    print(fd_array, alpha0_bright, beta0_bright)
    mf_amplitude_frame_bright = fixed_point_tracker(np.flip(fd_array, axis=0), params, c_matrices, alpha0=alpha0_bright,
                                                    beta0=beta0_bright)
    if mf_amplitude_frame_bright.dropna().shape[0] and not np.all(np.isclose(mf_amplitude_frame_bright.iloc[0].values, mf_amplitude_frame.iloc[fd_lower1_idx][['a_dim', 'b_dim']].values)):
        mf_amplitude_frame.loc[fd_lower1][['a_bright', 'b_bright']] = mf_amplitude_frame_bright.values[0, :]
        success = True
    else:
        success = False
    return mf_amplitude_frame, success



def map_mf(params, threshold=5e-5, check=False, fd_array=np.linspace(10.45, 10.49, 17)):
    print(threshold,'threshold')
    c_matrices = c_matrices_gen(params)
    mf_amplitude_frame = mf_characterise(params, fd_array, c_matrices=c_matrices)

    if mf_amplitude_frame['a_dim'].dropna().shape[0] == 0:
        return None

    while mf_amplitude_frame.dropna().shape[0] == 0:
        mf_amplitude_frame = find_overlap(mf_amplitude_frame, params)

    check_success = True
    while check_success:
        mf_amplitude_frame, check_success = check_lower(mf_amplitude_frame, params, c_matrices)

    indices = mf_amplitude_frame.index
    fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
    fd_lower1 = indices[fd_lower1_idx]
    df_lower = fd_lower2 - fd_lower1
    print('df_lower',df_lower)
    print('threshold',threshold)
    while df_lower > threshold:
        print(df_lower, 'Extending lower.')
        mf_amplitude_frame = extend_lower(mf_amplitude_frame, params)
        if check:
            check_success = True
            while check_success:
                mf_amplitude_frame, check_success = check_lower(mf_amplitude_frame, params, c_matrices)
        indices = mf_amplitude_frame.index
        fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
        fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
        fd_lower1 = indices[fd_lower1_idx]
        df_lower = fd_lower2 - fd_lower1


    check_success = True
    while check_success:
        mf_amplitude_frame, check_success = check_upper(mf_amplitude_frame, params, c_matrices)

    indices = mf_amplitude_frame.index
    fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
    fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
    fd_upper2 = indices[fd_upper2_idx]
    df_upper = fd_upper2 - fd_upper1
    while df_upper > threshold:
        print(df_upper, 'Extending upper')
        mf_amplitude_frame = extend_upper(mf_amplitude_frame, params)
        if check:
            check_success = True
            while check_success:
                mf_amplitude_frame, check_success = check_lower(mf_amplitude_frame, params, c_matrices)
        indices = mf_amplitude_frame.index
        fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
        fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
        fd_upper2 = indices[fd_upper2_idx]
        df_upper = fd_upper2 - fd_upper1

    return mf_amplitude_frame


def dalpha_calc_mf_duffing(alpha, params):
    dalpha = params.eps - 1j*(params.fc-params.fd)*alpha - 2*1j*params.chi*alpha*np.abs(alpha)**2 - 0.5*params.kappa*alpha
    return dalpha


def classical_eom_mf_duffing(x, params):
    alpha = x[0] + 1j * x[1]
    dalpha = dalpha_calc_mf_duffing(alpha, params)
    dx = np.array([dalpha.real, dalpha.imag])
    return dx


def locate_fixed_point_mf_duffing(params, alpha0=(0, 0)):
    x0 = np.array([alpha0[0], alpha0[1]])
    res = root(classical_eom_mf_duffing, x0, args=(params,), method='hybr')
    if res.success:
        alpha = res.x[0] + 1j * res.x[1]
    else:
        alpha = None
    return alpha


def fixed_point_tracker_duffing(fd_array, params, alpha0=0, fill_value=None, threshold=1e-4,
                                columns=['a'], crosscheck_frame=None):
    amplitude_array = np.zeros([fd_array.shape[0], 1], dtype=complex)
    trip = False
    for idx, fd in tqdm(enumerate(fd_array)):
        if not trip:
            params_instance = deepcopy(params)
            params_instance.fd = fd
            alpha_fixed = locate_fixed_point_mf_duffing(params_instance, alpha0=[alpha0.real, alpha0.imag])
            if alpha_fixed is None:
                # trip = True
                amplitude_array[idx, :] = [fill_value]
            else:
                # print(alpha_fixed)
                amplitude_array[idx, :] = [alpha_fixed]
                alpha0 = alpha_fixed
    amplitude_frame = pd.DataFrame(amplitude_array, index=fd_array, columns=columns)
    amplitude_frame.sort_index(inplace=True)
    return amplitude_frame


def mf_characterise_duffing(base_params, fd_array):
    alpha0 = 0
    mf_amplitude_frame_bright = fixed_point_tracker_duffing(np.flip(fd_array, axis=0), base_params, alpha0=alpha0)
    mf_amplitude_frame_dim = fixed_point_tracker_duffing(fd_array, base_params, alpha0=alpha0, columns=['a_dim'],
                                                         crosscheck_frame=mf_amplitude_frame_bright)
    mf_amplitude_frame_bright.columns = ['a_bright']
    mf_amplitude_frame = pd.concat([mf_amplitude_frame_bright, mf_amplitude_frame_dim], axis=1)

    overlap_find = ~np.isclose(mf_amplitude_frame['a_bright'].values, mf_amplitude_frame['a_dim'].values)

    if not np.any(overlap_find):
        mf_amplitude_frame = pd.DataFrame(mf_amplitude_frame.values[:, 0], index=mf_amplitude_frame.index,
                                          columns=['a'])
    else:
        start_idx = np.where(overlap_find)[0][0]
        end_idx = np.where(overlap_find)[0][-1]

        mf_amplitude_frame['a_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['a_dim'].iloc[end_idx + 1:] = None

    return mf_amplitude_frame


def compute_poly(beta, c):
    poly = 0
    for m, n, coeff in zip(c.row, c.col, c.data):
        poly += coeff * np.conjugate(beta) ** m * beta ** n
    return poly