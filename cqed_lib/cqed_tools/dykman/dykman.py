import numpy as np


def dykman_calc_single(params):
    omega_F = params['omega_F']
    epsilon = np.abs(params['eps'])
    omega_0 = params['omega_0']
    chi = params['chi']
    Gamma = params['kappa'] / 2
    gamma = 8 * omega_0 ** 2 * chi / 3
    n = params['n_c']
    # A = -2*epsilon*np.sqrt(2*omega_0)
    A = 2 * epsilon * np.sqrt(2 * omega_0)

    lam = 3 * gamma / (8 * omega_F ** 2 * (omega_F - omega_0))
    C = np.sqrt(8 * omega_F * (omega_F - omega_0) / (3 * gamma))
    E_sl = gamma * C ** 4

    delta_omega = omega_F - omega_0
    # kappa = Gamma / (E_sl * lam)
    kappa = Gamma / delta_omega

    beta = 3 * gamma * A ** 2 / (32 * omega_F ** 3 * (omega_F - omega_0) ** 3)

    A2_1 = (2 + (1 - 3 * kappa ** 2) ** 0.5) / 3.0
    A2_2 = (2 - (1 - 3 * kappa ** 2) ** 0.5) / 3.0

    beta_1 = (2.0 / 27) * (1 + 9 * kappa ** 2 - (1 - 3 * kappa ** 2) ** (1.5))
    beta_2 = (2.0 / 27) * (1 + 9 * kappa ** 2 + (1 - 3 * kappa ** 2) ** (1.5))

    b_1 = beta_1 ** 0.5 * (3 * A2_1 - 2) / (2 * kappa ** 2)
    b_2 = beta_2 ** 0.5 * (3 * A2_2 - 2) / (2 * kappa ** 2)

    eta_1 = beta - beta_1
    eta_2 = beta - beta_2
    # print(beta_1, b_1, eta_1)


    Omega_sw_1 = np.abs(delta_omega) * (b_1 * eta_1 / 2) ** 0.5 / (np.pi * beta_1 ** 0.25)
    Omega_sw_2 = np.abs(delta_omega) * (b_2 * eta_2 / 2) ** 0.5 / (np.pi * beta_2 ** 0.25)
    R_A_1 = 2 * np.sqrt(2) * np.abs(eta_1) ** 1.5 / (3 * kappa * np.abs(b_1) ** 0.5 * beta_1 ** 0.75 * (1 + 2 * n))
    R_A_2 = 2 * np.sqrt(2) * np.abs(eta_2) ** 1.5 / (3 * kappa * np.abs(b_2) ** 0.5 * beta_2 ** 0.75 * (1 + 2 * n))
    W_sw_1 = Omega_sw_1 * np.exp(R_A_1 / lam)
    W_sw_2 = Omega_sw_2 * np.exp(R_A_2 / lam)

    steps_dict = dict()
    steps_dict['beta_1'] = beta_1
    steps_dict['beta_2'] = beta_2
    steps_dict['kappa'] = kappa
    steps_dict['C'] = C
    steps_dict['lam'] = lam
    steps_dict['eta_1'] = eta_1
    steps_dict['eta_2'] = eta_2
    steps_dict['A2_1'] = A2_1
    steps_dict['A2_2'] = A2_2
    steps_dict['Omega_sw_1'] = Omega_sw_1
    steps_dict['Omega_sw_2'] = Omega_sw_2
    steps_dict['R_A_1'] = R_A_1
    steps_dict['R_A_2'] = R_A_2

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