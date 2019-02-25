import numpy as np


def dykman_calc_single(params, show_steps=False):

    omega_F = params['omega_F']
    epsilon = np.abs(params['eps'])
    omega_0 = params['omega_0']
    chi = params['chi']
    Gamma = params['kappa'] / 2
    gamma = 8 * omega_0 ** 2 * chi / 3
    n = params['n_c']

    if 'kappa_ph' not in params.keys():
        kappa_ph = 0.0
    else:
        kappa_ph = params['kappah_ph'] / 2

    A = 2 * epsilon * np.sqrt(2 * omega_0)

    lam = 3 * gamma / (8 * omega_F ** 2 * (omega_F - omega_0))
    chi_ph = kappa_ph / (lam * Gamma)

    # C = np.sqrt(8 * omega_F * (omega_F - omega_0) / (3 * gamma))
    # E_sl = gamma * C ** 4

    delta_omega = omega_F - omega_0

    Omega = delta_omega / Gamma
    beta_1 = (2.0 / 27) * (1 + 9 * Omega ** (-2) - (1 - 3 * Omega ** (-2)) ** (1.5))
    beta_2 = (2.0 / 27) * (1 + 9 * Omega ** (-2) + (1 - 3 * Omega ** (-2)) ** (1.5))
    Y_B_1 = (1.0 / 3.0) * (2 + (1 - 3 * Omega ** (-2)) ** (0.5))
    Y_B_2 = (1.0 / 3.0) * (2 - (1 - 3 * Omega ** (-2)) ** (0.5))
    D_B_1 = Omega ** (-1) * ((n + 0.5) + 0.5 * chi_ph * (1 - Y_B_1))
    D_B_2 = Omega ** (-1) * ((n + 0.5) + 0.5 * chi_ph * (1 - Y_B_2))
    beta = 3 * gamma * A ** 2 / (32 * omega_F ** 3 * (omega_F - omega_0) ** 3)

    b_1 = -(beta_1 ** 0.5) * (2 * Y_B_1) ** (-1) * (1 - 2 * (Omega ** 2) * Y_B_1 + Omega ** 2)
    b_2 = -(beta_2 ** 0.5) * (2 * Y_B_2) ** (-1) * (1 - 2 * (Omega ** 2) * Y_B_2 + Omega ** 2)

    eta_1 = beta - beta_1
    eta_2 = beta - beta_2

    steps_dict = dict()
    steps_dict['beta_1'] = beta_1
    steps_dict['beta_2'] = beta_2
    steps_dict['Omega'] = Omega
    steps_dict['lam'] = lam
    steps_dict['eta_1'] = eta_1
    steps_dict['eta_2'] = eta_2
    steps_dict['b_1'] = b_1
    steps_dict['b_2'] = b_2
    steps_dict['C_1'] = np.nan
    steps_dict['C_2'] = np.nan
    steps_dict['R_A_1'] = np.nan
    steps_dict['R_A_2'] = np.nan

    if eta_1 < 0 or eta_2 > 0:
        if show_steps:
            return np.nan, np.nan, steps_dict
        else:
            return np.nan, np.nan

    C_1 = np.abs(delta_omega) * (b_1 * eta_1 / 2) ** 0.5 / (np.pi * beta_1 ** 0.25)
    C_2 = np.abs(delta_omega) * (b_2 * eta_2 / 2) ** 0.5 / (np.pi * beta_2 ** 0.25)

    R_A_1 = np.sqrt(2) * (np.abs(eta_1) ** 1.5) / (3 * D_B_1 * (np.abs(b_1) ** 0.5) * (beta_1 ** 0.75))
    R_A_2 = np.sqrt(2) * (np.abs(eta_2) ** 1.5) / (3 * D_B_2 * (np.abs(b_2) ** 0.5) * (beta_2 ** 0.75))

    W_sw_1 = C_1 * np.exp(R_A_1 / lam)
    W_sw_2 = C_2 * np.exp(R_A_2 / lam)

    steps_dict['C_1'] = C_1
    steps_dict['C_2'] = C_2
    steps_dict['R_A_1'] = R_A_1
    steps_dict['R_A_2'] = R_A_2

    if show_steps:
        return W_sw_1, W_sw_2, steps_dict
    else:
        return W_sw_1, W_sw_2


dykman_calc = np.vectorize(dykman_calc_single)


def beta_calc_df(omega_F, params):
    Gamma = params['kappa'] / 2
    omega_0 = params['omega_0']
    delta_omega = omega_F - omega_0
    kappa = Gamma / delta_omega
    return beta_calc(kappa)


def beta_calc(kappa):
    beta_p = (1 + 9 * kappa ** 2 + (1 - 3 * kappa ** 2) ** 1.5) * 2.0 / 27.0
    beta_m = (1 + 9 * kappa ** 2 - (1 - 3 * kappa ** 2) ** 1.5) * 2.0 / 27.0
    return beta_m, beta_p


def eps_calc_df(omega_F, params):
    beta_m, beta_p = beta_calc_df(omega_F, params)
    omega_0 = params['omega_0']
    chi = params['chi']
    gamma = 8 * omega_0 ** 2 * chi / 3
    A_m = np.sqrt((32 * omega_F ** 3 * (omega_F - omega_0) ** 3 * beta_m) / (3 * gamma))
    A_p = np.sqrt((32 * omega_F ** 3 * (omega_F - omega_0) ** 3 * beta_p) / (3 * gamma))
    eps_m = A_m / (2 * np.sqrt(2 * omega_0))
    eps_p = A_p / (2 * np.sqrt(2 * omega_0))
    return eps_m, eps_p