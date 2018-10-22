import numpy as np
from qutip import *
from ..simulation.hamiltonian import coupling_calc, transmon_energies_calc


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


def transmon_hamiltonian_gen_mf(params):
    energies = transmon_energies_calc(params)
    transmon_hamiltonian = 0
    for n, energy in enumerate(energies):
        transmon_hamiltonian += (energy-n*params.fd)*fock_dm(params.t_levels, n)
    return transmon_hamiltonian


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