from qutip import *
from ..mf import *
import pandas as pd


def ham_gen_jc(params, alpha=0):
    sz = tensor(sigmaz(), qeye(params.c_levels))
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels)) + alpha
    ham = (params.fc-params.fd)*a.dag()*a
    ham += params.eps*(a+a.dag())
    ham += 0.5*(params.f01-params.fd)*sz
    ham += params.g*(a*sm.dag() + a.dag()*sm)
    ham *= 2*np.pi
    return ham


def c_ops_gen_jc(params, alpha=0):
    c_ops = []
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels)) + alpha
    if params.gamma > 0.0:
        c_ops.append(np.sqrt(2*np.pi*params.gamma)*sm)
        if params.n_t > 0:
            c_ops.append(np.sqrt(2*np.pi*params.gamma*(1+params.n_t))*sm.dag())
    if params.gamma_phi > 0.0:
        c_ops.append(np.sqrt(2*np.pi*params.gamma_phi)*sm.dag()*sm)
    if params.kappa > 0.0:
        c_ops.append(np.sqrt(2*np.pi*params.kappa)*a)
        if params.n_c > 0:
            c_ops.append(np.sqrt(2*np.pi*params.gamma*(1+params.n_c))*a.dag())
    return c_ops


class Spectrum:
    def __init__(self, parameters):
        print('hello')
        self.parameters = deepcopy(parameters)
        self.mf_amplitude = None
        self.me_amplitude = None
        self.transmission_exp = None

    def mf_calculate(self, fd_array):
        if self.mf_amplitude is None:
            self.mf_amplitude = map_mf_jc(self.parameters, fd_array=fd_array)
        else:
            fd0 = fd_array[0]
            fd1 = fd_array[-1]
            idx0 = self.mf_amplitude.index.get_loc(fd0, method='nearest')
            idx1 = self.mf_amplitude.index.get_loc(fd1, method='nearest')
            alpha0_dim = self.mf_amplitude['a_dim'].iloc[idx0]
            sm0_dim = self.mf_amplitude['sm_dim'].iloc[idx0]
            sz0_dim = self.mf_amplitude['sz_dim'].iloc[idx0]
            alpha0_bright = self.mf_amplitude['a_bright'].iloc[idx1]
            sm0_bright = self.mf_amplitude['sm_bright'].iloc[idx1]
            sz0_bright = self.mf_amplitude['sz_bright'].iloc[idx1]
            mf_amplitude_new = mf_characterise_jc(self.parameters, fd_array, alpha0_bright=alpha0_bright,
                                                  sm0_bright=sm0_bright, sz0_bright=sz0_bright, alpha0_dim=alpha0_dim,
                                                  sm0_dim=sm0_dim, sz0_dim=sz0_dim)
            self.mf_amplitude = pd.concat([self.mf_amplitude, mf_amplitude_new])

        self.mf_amplitude = self.mf_amplitude.sort_index()
        self.mf_amplitude = self.mf_amplitude[~self.mf_amplitude.index.duplicated(keep='first')]

    def generate_hilbert_params(self, c_levels_bi_scale=1.0, scale=0.5, fd_lower=None, fd_upper=None, max_shift=True,
                                c_levels_mono=10):
        print('generating')
        self.hilbert_params = generate_hilbert_params(self.mf_amplitude, c_levels_bi_scale=c_levels_bi_scale,
                                                      scale=scale, fd_lower=fd_lower, fd_upper=fd_upper,
                                                      max_shift=max_shift, c_levels_mono=c_levels_mono)

    def me_calculate(self, solver_kwargs={}, c_levels_bi_scale=1.0, scale=0.5, fd_lower=None, fd_upper=None,
                     max_shift=True, c_levels_mono=10):
        self.generate_hilbert_params(c_levels_bi_scale=c_levels_bi_scale, scale=scale, max_shift=max_shift,
                                     c_levels_mono=c_levels_mono)
        # self.hilbert_params = generate_hilbert_params(self.mf_amplitude, c_levels_bi_scale=c_levels_bi_scale, scale=scale, fd_lower=fd_lower, fd_upper=fd_upper, max_shift=max_shift)
        # self.generate_hilbert_params(c_levels_bi_scale=c_levels_bi_scale, scale=scale)


        a_array = np.zeros(self.mf_amplitude.shape[0], dtype=complex)

        params = deepcopy(self.parameters)
        for fd_idx, fd, alpha0, c_levels in tqdm(
                zip(np.arange(self.hilbert_params.index.shape[0]), self.hilbert_params.index,
                    self.hilbert_params['alpha_0'], self.hilbert_params['c_levels'])):
            params.fd = fd
            params.c_levels = c_levels
            ham = ham_gen_jc(params, alpha=alpha0)
            c_ops = c_ops_gen_jc(params, alpha=alpha0)
            try:
                rho = steadystate(ham, c_ops, **solver_kwargs)
                a = tensor(qeye(2), destroy(params.c_levels)) + alpha0
                a_array[fd_idx] = expect(rho, a)
            except:
                print('Failure at fd = ' + str(fd))
                a_array[fd_idx] = np.nan

        self.me_amplitude = pd.DataFrame(a_array, index=self.mf_amplitude.index)

    def plot(self, axes=None, mf=True, me=True, db=True):

        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.set_xlabel(r'$f_d$ (GHz)')
            axes.set_ylabel(r'|$\langle a \rangle$|')

        if db:
            if me:
                if self.me_amplitude is not None:
                    axes.plot(self.me_amplitude.dropna().index, 20 * np.log10(np.abs(self.me_amplitude.dropna())),
                              marker='o')
            if mf:
                if self.mf_amplitude.shape[1] == 1:
                    axes.plot(self.mf_amplitude.index, 20 * np.log10(np.abs(self.mf_amplitude['a'])), marker='o')
                else:
                    axes.plot(self.mf_amplitude.index, 20 * np.log10(np.abs(self.mf_amplitude['a_bright'])), marker='o')
                    axes.plot(self.mf_amplitude.index, 20 * np.log10(np.abs(self.mf_amplitude['a_dim'])), marker='o')
        else:
            if me:
                if self.me_amplitude is not None:
                    axes.plot(self.me_amplitude.dropna().index, np.abs(self.me_amplitude.dropna()), marker='o')
            if mf:
                if self.mf_amplitude.shape[1] == 1:
                    axes.plot(self.mf_amplitude.index, np.abs(self.mf_amplitude['a']), marker='o')
                else:
                    axes.plot(self.mf_amplitude.index, np.abs(self.mf_amplitude['a_bright']), marker='o')
                    axes.plot(self.mf_amplitude.index, np.abs(self.mf_amplitude['a_dim']), marker='o')

    def plot_transmission(self, axes=None, scale=4.851024710399999e-09, exp=True, sim=True):

        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes.set_ylabel(r'$T_{NA}$ (dB)')
            axes.set_xlabel(r'$f_{df}$ (GHz)')

        if sim and self.me_amplitude is not None:
            self.transmission = scale * np.abs(self.me_amplitude.dropna()) ** 2 / self.parameters.eps ** 2
            axes.plot(self.transmission.index, 10 * np.log10(self.transmission), marker='o', label='Sim')

        if exp and self.transmission_exp is not None:
            axes.plot(self.transmission_exp.index, self.transmission_exp, label='Exp')

    def load_exp(self, path):
        self.transmission_exp = pd.read_csv(path, dtype=float, header=None).T
        self.transmission_exp = self.transmission_exp.set_index(0)


def generate_hilbert_params(mf_amplitude, fd_lower=None, fd_upper=None, scale=0.5, c_levels_mono=10,
                            c_levels_bi_scale=1.0, max_shift=True):
    if 'a_dim' not in mf_amplitude.columns:

        hilbert_params = deepcopy(mf_amplitude)
        hilbert_params.columns = ['alpha_0']
        hilbert_params['c_levels'] = params.c_levels

    else:

        mf_amplitude_bistable = mf_amplitude.dropna().loc[fd_lower:fd_upper]

        alpha_0_monostable_bright = mf_amplitude['a_bright'].dropna().drop(labels=mf_amplitude_bistable.index).loc[
                                    fd_upper:]
        alpha_0_monostable_dim = mf_amplitude['a_dim'].dropna().drop(labels=mf_amplitude_bistable.index).loc[:fd_lower]
        alpha_diff_bistable = mf_amplitude_bistable['a_bright'] - mf_amplitude_bistable['a_dim']
        alpha_diff_bistable_min = np.min(np.abs(alpha_diff_bistable))
        alpha_dim_bistable = mf_amplitude_bistable['a_dim']
        alpha_diff_bistable_unit = alpha_diff_bistable / np.abs(alpha_diff_bistable)
        if max_shift:
            alpha_0_bistable = alpha_dim_bistable + scale * alpha_diff_bistable_min * alpha_diff_bistable_unit
        else:
            alpha_0_bistable = alpha_dim_bistable + scale * alpha_diff_bistable

        hilbert_params_mono = pd.concat(
            [alpha_0_monostable_bright.to_frame('alpha_0'), alpha_0_monostable_dim.to_frame('alpha_0')])
        hilbert_params_mono['c_levels'] = c_levels_mono

        hilbert_params_bi = alpha_0_bistable.to_frame('alpha_0')
        hilbert_params_bi['c_levels'] = int(np.ceil(c_levels_bi_scale * alpha_diff_bistable_min ** 2))

        hilbert_params = pd.concat([hilbert_params_mono, hilbert_params_bi])
        hilbert_params = hilbert_params.sort_index()

    return hilbert_params