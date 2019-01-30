import numpy as np
from qutip import *
from scipy.special import factorial
from scipy import sparse
from ..simulation.hamiltonian_gen import coupling_calc, transmon_energies_calc


def hamiltonian_mf(params, alpha, transmon=True, duffing=False):
    H = (params.fc - params.fd) * np.conjugate(alpha) * alpha + params.eps * (alpha + np.conjugate(alpha))
    if transmon is True:
        if duffing:
            b = destroy(params.t_levels)
            H += (params.f01 - params.fd)*b.dag()*b + params.g*(alpha*b.dag() + np.conjugate(alpha)*b) + 0.5*params.u*b.dag()*b.dag()*b*b
        else:
            transmon_hamiltonian = transmon_hamiltonian_gen_mf(params)
            coupling_hamiltonian = coupling_hamiltonian_gen_mf(params, alpha)
            H += transmon_hamiltonian + coupling_hamiltonian
    else:
        b = destroy(params.t_levels)
        H += (params.f01 - params.fd)*b.dag()*b + params.g*(alpha*b.dag() + np.conjugate(alpha)*b)
    return H


def collapse_operators_mf(params):
    sm = destroy(params.t_levels)
    c_ops = []
    if params.gamma != 0:
        c_ops.append(np.sqrt(params.gamma*(params.n_t+1)) * sm)
        if params.n_t != 0:
            c_ops.append(np.sqrt(params.gamma*params.n_t) * sm.dag())
    if params.gamma_phi != 0:
        dispersion_op = sm.dag()*sm
        c_ops.append(np.sqrt(params.gamma_phi)*dispersion_op)
    return c_ops


def c_matrices_gen(params, duffing=False, threshold=1e-10):
    b = destroy(params.t_levels)

    if duffing:
        ham_0 = params.f01 * b.dag() * b + 0.5 * params.u * b.dag() * b.dag() * b * b
        down_coupling = params.g * b
    else:
        ham_0 = transmon_hamiltonian_gen_mf(params, fd=0)
        down_coupling = down_coupling_gen(params)

    c_ops = collapse_operators_mf(params)

    rho_dot_0 = -1j * commutator(b, ham_0)
    for c_op in c_ops:
        rho_dot_0 += c_op.dag() * b * c_op - 0.5 * b * c_op.dag() * c_op - 0.5 * c_op.dag() * c_op * b

    c_0 = normal_order_calc(rho_dot_0, threshold=threshold)

    rho_dot_alpha = -1j * commutator(b, down_coupling.dag())
    c_alpha = normal_order_calc(rho_dot_alpha, threshold=threshold)
    rho_dot_alpha_conj = -1j * commutator(b, down_coupling)
    c_alpha_conj = normal_order_calc(rho_dot_alpha_conj, threshold=threshold)

    c_down = normal_order_calc(down_coupling, threshold=threshold)

    c_matrices = {'c_0': c_0,
                  'c_alpha': c_alpha,
                  'c_alpha_conj': c_alpha_conj,
                  'c_down': c_down}

    return c_matrices


def normal_order_calc(operator, limit=None, threshold=1e-10):
    assert len(operator.dims[0]) == 1
    levels = operator.dims[0][0]
    if limit is None:
        limit = levels
    c = np.zeros([limit, limit], dtype=complex)
    for i in range(limit):
        for j in range(limit):
            A = (basis(levels, i).dag() * operator * basis(levels, j)).full()[0, 0] / np.sqrt(
                factorial(i) * factorial(j))
            for d in range(1, min(i, j) + 1):
                A -= c[i - d, j - d] / factorial(d)
            c[i, j] = A
    c *= np.abs(c) > threshold
    c = sparse.coo_matrix(c)
    return c


def transmon_hamiltonian_gen_mf(params, fd=None):
    energies = transmon_energies_calc(params)
    transmon_hamiltonian = 0
    for n, energy in enumerate(energies):
        if fd is not None:
            transmon_hamiltonian += (energy-n*fd)*fock_dm(params.t_levels, n)
        else:
            transmon_hamiltonian += (energy-n*params.fd)*fock_dm(params.t_levels, n)
    return transmon_hamiltonian


def coupling_hamiltonian_gen_mf(params, alpha):
    lower_levels = np.arange(0,params.t_levels-1)
    upper_levels = np.arange(1,params.t_levels)
    q = -params.Ej / (2 * params.Ec)
    coupling_array = coupling_calc(lower_levels,upper_levels,q)
    coupling_array = coupling_array/coupling_array[0]
    down_transmon_transitions = 0
    for i, coupling in enumerate(coupling_array):
        down_transmon_transitions += coupling*basis(params.t_levels,i)*basis(params.t_levels,i+1).dag()
    down_transmon_transitions *= np.conjugate(alpha)
    coupling_hamiltonian = down_transmon_transitions + down_transmon_transitions.dag()
    coupling_hamiltonian *= params.g
    return coupling_hamiltonian


def down_coupling_gen(params):
    lower_levels = np.arange(0,params.t_levels-1)
    upper_levels = np.arange(1,params.t_levels)
    q = -params.Ej / (2 * params.Ec)
    coupling_array = coupling_calc(lower_levels,upper_levels,q)
    coupling_array = coupling_array/coupling_array[0]
    down_transmon_transitions = 0
    for i, coupling in enumerate(coupling_array):
        down_transmon_transitions += coupling*basis(params.t_levels,i)*basis(params.t_levels,i+1).dag()
    down_transmon_transitions *= params.g
    return down_transmon_transitions