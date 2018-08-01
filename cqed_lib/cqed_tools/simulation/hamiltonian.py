import scipy.integrate
from scipy.special import factorial
from qutip import *
import numpy as np
import scipy.integrate
from .gsl import *


class Parameters:
    def __init__(self, fc=None, Ej=None, g=None, Ec=None, eps=None, fd=None, kappa=None, gamma=None, t_levels=None, c_levels=None, gamma_phi=None, kappa_phi=None, n_t=None, n_c=None, f01=None, chi=None, ntraj=None):
        self.fc = fc
        self.Ej = Ej
        self.eps = eps
        self.g = g
        self.Ec = Ec
        self.gamma = gamma
        self.kappa = kappa
        self.t_levels = t_levels
        self.c_levels = c_levels
        self.fd = fd
        self.gamma_phi = gamma_phi
        self.kappa_phi = kappa_phi
        self.n_t = n_t
        self.n_c = n_c
        self.f01 = f01
        self.chi = chi
        self.ntraj = ntraj
        self.labels = ['f_d', 'eps', 'E_j', 'f_c', 'g', 'kappa', 'kappa_phi', 'gamma', 'gamma_phi', 'E_c', 'n_t', 'n_c', 'f01', 'chi']

    def copy(self):
        params = Parameters(self.fc, self.Ej, self.g, self.Ec, self.eps, self.fd, self.kappa, self.gamma, self.t_levels, self.c_levels, self.gamma_phi, self.kappa_phi, self.n_t, self.n_c, self.f01, self.chi)
        return params


class ParametersPartial:
    def __init__(self, fc=None, Ej=None, g=None, Ec=None, eps=None, fd=None, kappa=None, gamma=None, t_levels=None, c_levels=None, gamma_phi=None, kappa_phi=None, n_t=None, n_c=None):
        self.fc = fc
        self.Ej = Ej
        self.eps = eps
        self.g = g
        self.Ec = Ec
        self.gamma = gamma
        self.kappa = kappa
        self.t_levels = t_levels
        self.c_levels = c_levels
        self.fd = fd
        self.gamma_phi = gamma_phi
        self.kappa_phi = kappa_phi
        self.n_t = n_t
        self.n_c = n_c
        self.labels = ['f_d', 'eps', 'E_j', 'f_c', 'g', 'kappa', 'kappa_phi', 'gamma', 'gamma_phi', 'E_c', 'n_t', 'n_c']

    def copy(self):
        params = ParametersPartial(fc=self.fc, Ej=self.Ej, g=self.g, Ec=self.Ec, eps=self.eps, fd=self.fd, kappa=self.kappa, gamma=self.gamma, t_levels=self.t_levels, c_levels=self.c_levels, gamma_phi=self.gamma_phi, kappa_phi=self.kappa_phi, n_t=self.n_t, n_c=self.n_c)
        return params



def collapse_operators(params):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = []
    if params.kappa != 0:
        c_ops.append(np.sqrt(params.kappa*(params.n_c+1)) * a)
        if params.n_c != 0:
            c_ops.append(np.sqrt(params.kappa*params.n_c) * a.dag())
    if params.gamma != 0:
        c_ops.append(np.sqrt(params.gamma*(params.n_t+1)) * sm)
        if params.n_t != 0:
            c_ops.append(np.sqrt(params.gamma*params.n_t) * sm.dag())
    if params.gamma_phi != 0:
        #dispersion_op = dispersion_op_gen(params)
        dispersion_op = sm.dag()*sm
        c_ops.append(np.sqrt(params.gamma_phi)*dispersion_op)
    return c_ops


def charge_dispersion_calc(level, Ec, Ej):
    dispersion = (-1)**level * Ec * 2**(4*level+5) * np.sqrt(2.0/np.pi) * (Ej/(2*Ec))**(level/2.0+3/4.0) * np.exp(-np.sqrt(8*Ej/Ec))
    dispersion /= factorial(level)
    return dispersion


def transmon_params_calc(sys_params):
    alpha = 2*sys_params.chi
    Ec = -alpha
    Ej = (Ec/8)*(sys_params.fq/alpha)**2
    return Ec, Ej


def dispersion_op_gen(sys_params):
    Ec, Ej = transmon_params_calc(sys_params)
    normalization = charge_dispersion_calc(1,Ec,Ej) - charge_dispersion_calc(0,Ec,Ej)
    dispersion_op = 0
    for i in range(sys_params.t_levels):
        dispersion_op += fock_dm(sys_params.t_levels, i)*charge_dispersion_calc(i,Ec,Ej)
    dispersion_op /= normalization
    dispersion_op = tensor(qeye(sys_params.c_levels), dispersion_op)
    return dispersion_op


def mathieu_ab_single(idx, q):
    if idx % 2 == 0:
        characteristic = mathieu_a(idx, q)
    else:
        characteristic = mathieu_b(idx+1, q)
    return characteristic

mathieu_ab = np.vectorize(mathieu_ab_single)


def transmon_energies_calc(params, normalize=True):
    Ec = params.Ec
    Ej = params.Ej
    q = -Ej/(2*Ec)
    n_levels = params.t_levels
    energies = Ec*mathieu_ab(np.arange(n_levels),q)
    ref_energies = energies - energies[0]
    if normalize:
        return ref_energies
    else:
        return energies


def transmon_hamiltonian_gen(params):
    energies = transmon_energies_calc(params)
    transmon_hamiltonian = 0
    for n, energy in enumerate(energies):
        transmon_hamiltonian += (energy-n*params.fd)*fock_dm(params.t_levels, n)
    transmon_hamiltonian = tensor(qeye(params.c_levels), transmon_hamiltonian)
    return transmon_hamiltonian


def transition_func(theta, i, j, q):
    bra = np.conjugate(psi_calc(theta,i,q))
    step = 1e-7
    ket = derivative_calc(psi_calc,theta,[j,q],step)
    overlap_point = bra * ket
    return overlap_point


def overlap_func(theta, i, j, q):
    bra = np.conjugate(psi_calc(theta,i,q))
    ket = psi_calc(theta,j,q)
    overlap_point = bra * ket
    return overlap_point


def coupling_calc_single(i, j, q):
    coupling, error = scipy.integrate.quad(transition_func, 0, 2*np.pi, args=(i, j, q))
    return np.abs(coupling)


coupling_calc = np.vectorize(coupling_calc_single)


def psi_calc(theta, idx, q):
    if idx % 2 == 0:
        psi = mathieu_ce(idx,q,theta/2)/np.sqrt(np.pi)
    else:
        psi = mathieu_se(idx+1,q,theta/2)/np.sqrt(np.pi)
    return psi


def derivative_calc(func, x, params, step):
    derivative = (func(x+step,*params)-func(x,*params))/step
    return derivative


def low_coupling(idx,q):
    coupling = np.sqrt((idx+1)/2.0) * (-q/4)**0.25
    return coupling


def low_energies_calc_single(idx, params):
    Ec = params.Ec
    Ej = params.Ej
    energy = -Ej + np.sqrt(8.0*Ec*Ej)*(idx+0.5) - Ec*(6.0*idx**2 + 6.0*idx +3.0)/12.0
    return energy

low_energies_calc = np.vectorize(low_energies_calc_single)


def high_energies_calc_single(idx, params):
    if idx % 2 == 0:
        energy = params.Ec * idx**2
    else:
        energy = params.Ec * (idx+1)**2
    return energy


high_energies_calc = np.vectorize(high_energies_calc_single)


def coupling_hamiltonian_gen(params):
    lower_levels = np.arange(0,params.t_levels-1)
    upper_levels = np.arange(1,params.t_levels)
    q = -params.Ej / (2 * params.Ec)
    coupling_array = coupling_calc(lower_levels,upper_levels,q)
    coupling_array = coupling_array/coupling_array[0]
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    down_transmon_transitions = 0
    for i, coupling in enumerate(coupling_array):
        down_transmon_transitions += coupling*basis(params.t_levels,i)*basis(params.t_levels,i+1).dag()
    down_transmon_transitions = tensor(qeye(params.c_levels), down_transmon_transitions)
    down_transmon_transitions *= a.dag()
    coupling_hamiltonian = down_transmon_transitions + down_transmon_transitions.dag()
    coupling_hamiltonian *= params.g
    return coupling_hamiltonian


def hamiltonian(params, transmon=True, alpha=0, beta=0):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    H = (params.fc - params.fd) * a.dag() * a + params.eps * (a + a.dag())
    if transmon is True:
        transmon_hamiltonian = transmon_hamiltonian_gen(params)
        coupling_hamiltonian = coupling_hamiltonian_gen(params)
        H += transmon_hamiltonian + coupling_hamiltonian
    else:
        b = tensor(qeye(params.c_levels), destroy(params.t_levels))
        H += (params.f01 - params.fd)*b.dag()*b + params.g*(a*b.dag() + a.dag()*b)
    diplacement = tensor(displace(params.c_levels, alpha), displace(params.t_levels, beta))
    return displacement.dag()*H*displacement


def hamiltonian_eliminated(params):
    b = tensor(qeye(params.c_levels), destroy(params.t_levels))
    dims = b.dims
    transmon_linear = (params.f01-params.fd)*b.dag()*b
    transmon_nonlinear = params.chi*(b.dag()*b.dag())*(b*b)
    transmon_hamiltonian = transmon_linear + transmon_nonlinear
    H = transmon_hamiltonian
    H.dims = dims
    H += params.eps*b.dag() + np.conjugate(params.eps)*b
    return H