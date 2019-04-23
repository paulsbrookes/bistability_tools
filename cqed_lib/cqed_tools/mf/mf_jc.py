import numpy as np
from qutip import *
from scipy.optimize import root
from cqed_tools.mf.hamiltonian_gen_mf import hamiltonian_mf, collapse_operators_mf, c_matrices_gen
from copy import deepcopy
import pandas as pd
from tqdm import tqdm


def dalpha_calc_mf_jc(alpha, sm, sz, params):
    dalpha = -(0.5*params.kappa + 1j*(params.fc-params.fd))*alpha - 1j*params.eps - 1j*params.g*sm
    return dalpha


def dsm_calc_mf_jc(alpha, sm, sz, params):
    dsm = -(0.5*params.gamma + 1j*(params.f01-params.fd))*sm + 1j*params.g*alpha*sz
    return dsm


def dsz_calc_mf_jc(alpha, sm, sz, params):
    dsz = -params.gamma*(sz+1) - 2*1j*params.g*(alpha*np.conjugate(sm)-np.conjugate(alpha)*sm)
    return dsz


def classical_eom_mf_jc(x, params):
    alpha = x[0] + 1j * x[1]
    sm = x[2] + 1j * x[3]
    sz = x[4]
    dalpha = dalpha_calc_mf_jc(alpha, sm, sz, params)
    dsm = dsm_calc_mf_jc(alpha, sm, sz, params)
    dsz = dsz_calc_mf_jc(alpha, sm, sz, params)
    dx = np.array([dalpha.real, dalpha.imag, dsm.real, dsm.imag, dsz.real])
    return dx


def locate_fixed_point_mf_jc(params, alpha0=(0, 0), sm0=(0, 0), sz0=0):
    x0 = np.array([alpha0[0], alpha0[1], sm0[0], sm0[1], sz0])
    res = root(classical_eom_mf_jc, x0, args=(params), method='hybr')
    if res.success:
        alpha = res.x[0] + 1j * res.x[1]
        sm = res.x[2] + 1j * res.x[3]
        sz = res.x[4]
    else:
        alpha, sm, sz = None, None, None
    return alpha, sm, sz


def fixed_point_tracker_jc(fd_array, params, alpha0=0, sm0=0, sz0=0, fill_value=None,
                        columns=['a', 'sm', 'sz']):
    amplitude_array = np.zeros([fd_array.shape[0], 3], dtype=complex)
    trip = False
    found_first_point = False
    for idx, fd in tqdm(enumerate(fd_array)):
        if not trip:
            params_instance = deepcopy(params)
            params_instance.fd = fd
            alpha_fixed, sm_fixed, sz_fixed = locate_fixed_point_mf_jc(params_instance,
                                                            alpha0=[alpha0.real, alpha0.imag],
                                                            sm0=[sm0.real, sm0.imag],
                                                            sz0=sz0.real)
            if alpha_fixed is None:
                if found_first_point == True:
                    trip = True
                amplitude_array[idx, :] = [fill_value, fill_value, fill_value]
            else:
                found_first_point = True
                amplitude_array[idx, :] = [alpha_fixed, sm_fixed, sz_fixed]
                alpha0 = alpha_fixed
                sm0 = sm_fixed
                sz0 = sz_fixed
        else:
            amplitude_array[idx, :] = [fill_value, fill_value, fill_value]
    amplitude_frame = pd.DataFrame(amplitude_array, index=fd_array, columns=columns)
    amplitude_frame.sort_index(inplace=True)
    return amplitude_frame


def mf_characterise_jc(base_params, fd_array, alpha0_bright=0, sm0_bright=0, sz0_bright=0, alpha0_dim=0, sm0_dim=0, sz0_dim=0):
    mf_amplitude_frame_bright = fixed_point_tracker_jc(np.flip(fd_array, axis=0), base_params, alpha0=alpha0_bright, sm0=sm0_bright, sz0=sz0_bright)
    mf_amplitude_frame_dim = fixed_point_tracker_jc(fd_array, base_params, alpha0=alpha0_dim, sm0=sm0_dim, sz0=sz0_dim,
                                                 columns=['a_dim', 'sm_dim', 'sz_dim'])
    mf_amplitude_frame_bright.columns = ['a_bright', 'sm_bright', 'sz_bright']
    mf_amplitude_frame = pd.concat([mf_amplitude_frame_bright, mf_amplitude_frame_dim], axis=1)

    overlap_find = ~np.isclose(mf_amplitude_frame['a_bright'].values, mf_amplitude_frame['a_dim'].values)

    if not np.any(overlap_find):
        mf_amplitude_frame = pd.DataFrame(mf_amplitude_frame.values[:, 0], index=mf_amplitude_frame.index,
                                          columns=['a'])
    else:
        start_idx = np.where(overlap_find)[0][0]
        end_idx = np.where(overlap_find)[0][-1]

        mf_amplitude_frame['a_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['sm_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['sz_bright'].iloc[0:start_idx] = None
        mf_amplitude_frame['a_dim'].iloc[end_idx + 1:] = None
        mf_amplitude_frame['sm_dim'].iloc[end_idx + 1:] = None
        mf_amplitude_frame['sz_dim'].iloc[end_idx + 1:] = None

    return mf_amplitude_frame


def map_mf_jc(params, threshold=5e-5, check=False, fd_array=np.linspace(10.45, 10.49, 17)):
    print(threshold,'threshold')
    mf_amplitude_frame = mf_characterise_jc(params, fd_array)

    if mf_amplitude_frame['a_dim'].dropna().shape[0] == 0:
        return None

    while mf_amplitude_frame.dropna().shape[0] == 0:
        mf_amplitude_frame = find_overlap_jc(mf_amplitude_frame, params)

    check_success = True
    while check_success:
        mf_amplitude_frame, check_success = check_lower_jc(mf_amplitude_frame, params)

    indices = mf_amplitude_frame.index
    fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
    fd_lower1 = indices[fd_lower1_idx]
    df_lower = fd_lower2 - fd_lower1
    while df_lower > threshold:
        print(df_lower, 'Extending lower.')
        mf_amplitude_frame = extend_lower_jc(mf_amplitude_frame, params)
        if check:
            check_success = True
            while check_success:
                mf_amplitude_frame, check_success = check_lower_jc(mf_amplitude_frame, params)
        indices = mf_amplitude_frame.index
        fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
        fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
        fd_lower1 = indices[fd_lower1_idx]
        df_lower = fd_lower2 - fd_lower1


    check_success = True
    while check_success:
        mf_amplitude_frame, check_success = check_upper_jc(mf_amplitude_frame, params)

    indices = mf_amplitude_frame.index
    fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
    fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
    fd_upper2 = indices[fd_upper2_idx]
    df_upper = fd_upper2 - fd_upper1
    while df_upper > threshold:
        print(df_upper, 'Extending upper')
        mf_amplitude_frame = extend_upper_jc(mf_amplitude_frame, params)
        if check:
            check_success = True
            while check_success:
                mf_amplitude_frame, check_success = check_lower_jc(mf_amplitude_frame, params)
        indices = mf_amplitude_frame.index
        fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
        fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
        fd_upper2 = indices[fd_upper2_idx]
        df_upper = fd_upper2 - fd_upper1

    return mf_amplitude_frame


def find_overlap_jc(mf_amplitude_frame, params):
    alpha0_dim = mf_amplitude_frame['a_dim'].dropna().iloc[-1]
    sm0_dim = mf_amplitude_frame['sm_dim'].dropna().iloc[-1]
    smz_dim = mf_amplitude_frame['sz_dim'].dropna().iloc[-1]
    fd_lower = mf_amplitude_frame['a_dim'].dropna().index[-1]
    alpha0_bright = mf_amplitude_frame['a_bright'].dropna().iloc[0]
    sm0_bright = mf_amplitude_frame['sm_bright'].dropna().iloc[0]
    sz0_bright = mf_amplitude_frame['sz_bright'].dropna().iloc[0]
    fd_upper = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_array = np.linspace(fd_lower, fd_upper, 5)[1:-1]
    new_mf_amplitude_frame = mf_characterise_jc(params, fd_array, alpha0_bright=alpha0_bright, sm0_bright=sm0_bright,
                                             alpha0_dim=alpha0_dim, sm0_dim=sm0_dim)
    combined = pd.concat([mf_amplitude_frame, new_mf_amplitude_frame])
    combined.sort_index(inplace=True)
    return combined


def extend_lower_jc(mf_amplitude_frame, params):
    indices = mf_amplitude_frame.index
    fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
    fd_lower1 = indices[fd_lower1_idx]
    fd_lower = 0.5 * (fd_lower1 + fd_lower2)
    alpha0_bright = mf_amplitude_frame['a_bright'].dropna().iloc[0]
    sm0_bright = mf_amplitude_frame['sm_bright'].dropna().iloc[0]
    sz0_bright = mf_amplitude_frame['sz_bright'].dropna().iloc[0]
    alpha0_dim = 0.5 * (mf_amplitude_frame['a_dim'].iloc[fd_lower1_idx] + mf_amplitude_frame['a_dim'].iloc[fd_lower1_idx + 1])
    sm0_dim = 0.5 * (mf_amplitude_frame['sm_dim'].iloc[fd_lower1_idx] + mf_amplitude_frame['sm_dim'].iloc[fd_lower1_idx + 1])
    sz0_dim = 0.5 * (mf_amplitude_frame['sz_dim'].iloc[fd_lower1_idx] + mf_amplitude_frame['sz_dim'].iloc[fd_lower1_idx + 1])
    fd_array = np.array([fd_lower])
    new_mf_amplitude_frame = mf_characterise_jc(params, fd_array, alpha0_bright=alpha0_bright, sm0_bright=sm0_bright, sz0_bright=sz0_bright,
                                             alpha0_dim=alpha0_dim, sm0_dim=sm0_dim, sz0_dim=sz0_dim)
    combined = pd.concat([mf_amplitude_frame, new_mf_amplitude_frame])
    combined.sort_index(inplace=True)
    return combined


def extend_upper_jc(mf_amplitude_frame, params):
    indices = mf_amplitude_frame.index
    fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
    fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
    fd_upper2 = indices[fd_upper2_idx]
    fd_upper = 0.5 * (fd_upper1 + fd_upper2)
    alpha0_dim = mf_amplitude_frame['a_dim'].dropna().iloc[-1]
    sm0_dim = mf_amplitude_frame['sm_dim'].dropna().iloc[-1]
    sz0_dim = mf_amplitude_frame['sz_dim'].dropna().iloc[-1]
    alpha0_bright = 0.5 * (
    mf_amplitude_frame['a_bright'].iloc[fd_upper2_idx] + mf_amplitude_frame['a_bright'].iloc[fd_upper2_idx - 1])
    sm0_bright = 0.5 * (
    mf_amplitude_frame['sm_bright'].iloc[fd_upper2_idx] + mf_amplitude_frame['sm_bright'].iloc[fd_upper2_idx - 1])
    sz0_bright = 0.5 * (
    mf_amplitude_frame['sz_bright'].iloc[fd_upper2_idx] + mf_amplitude_frame['sz_bright'].iloc[fd_upper2_idx - 1])
    fd_array = np.array([fd_upper])
    new_mf_amplitude_frame = mf_characterise_jc(params, fd_array, alpha0_bright=alpha0_bright, sm0_bright=sm0_bright, sz0_bright=sz0_bright,
                                             alpha0_dim=alpha0_dim, sm0_dim=sm0_dim, sz0_dim=sz0_dim)
    combined = pd.concat([mf_amplitude_frame, new_mf_amplitude_frame])
    combined.sort_index(inplace=True)
    return combined


def check_upper_jc(mf_amplitude_frame, params):
    indices = mf_amplitude_frame.index
    fd_upper1 = mf_amplitude_frame['a_dim'].dropna().index[-1]
    fd_upper2_idx = np.argwhere(indices == fd_upper1)[0, 0] + 1
    fd_upper2 = indices[fd_upper2_idx]
    alpha0_dim = mf_amplitude_frame['a_dim'].iloc[fd_upper2_idx - 1]
    sm0_dim = mf_amplitude_frame['sm_dim'].iloc[fd_upper2_idx - 1]
    sz0_dim = mf_amplitude_frame['sz_dim'].iloc[fd_upper2_idx - 1]
    fd_array = np.array([fd_upper2])
    mf_amplitude_frame_dim = fixed_point_tracker_jc(np.flip(fd_array, axis=0), params, alpha0=alpha0_dim, sm0=sm0_dim, sz0=sz0_dim)
    if mf_amplitude_frame_dim.dropna().shape[0] and not np.all(np.isclose(mf_amplitude_frame_dim.iloc[0].values, mf_amplitude_frame.iloc[fd_upper2_idx][['a_bright', 'b_bright']].values)):
        mf_amplitude_frame.loc[fd_upper2][['a_dim', 'b_dim']] = mf_amplitude_frame_dim.values[0, :]
        success = True
    else:
        success = False
    return mf_amplitude_frame, success


def check_lower_jc(mf_amplitude_frame, params):
    indices = mf_amplitude_frame.index
    fd_lower2 = mf_amplitude_frame['a_bright'].dropna().index[0]
    fd_lower1_idx = np.argwhere(indices == fd_lower2)[0, 0] - 1
    fd_lower1 = indices[fd_lower1_idx]
    alpha0_bright = mf_amplitude_frame['a_bright'].iloc[fd_lower1_idx + 1]
    sm0_bright = mf_amplitude_frame['sm_bright'].iloc[fd_lower1_idx + 1]
    sz0_bright = mf_amplitude_frame['sz_bright'].iloc[fd_lower1_idx + 1]
    fd_array = np.array([fd_lower1])
    mf_amplitude_frame_bright = fixed_point_tracker_jc(np.flip(fd_array, axis=0), params, alpha0=alpha0_bright,
                                                    sm0=sm0_bright, sz0=sz0_bright)
    if mf_amplitude_frame_bright.dropna().shape[0] and not np.all(np.isclose(mf_amplitude_frame_bright.iloc[0].values, mf_amplitude_frame.iloc[fd_lower1_idx][['a_dim', 'b_dim']].values)):
        mf_amplitude_frame.loc[fd_lower1][['a_bright', 'b_bright']] = mf_amplitude_frame_bright.values[0, :]
        success = True
    else:
        success = False
    return mf_amplitude_frame, success