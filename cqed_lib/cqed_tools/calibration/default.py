from .spectra_tools import Parameters

class DefaultParameters(Parameters):
    def __init__(self):
        Ec = 0.2197428476508422
        fc = 10.4263
        Ej = 46.813986415542402
        g = 0.28492608868781633
        eps = 0.0
        gamma = 0.0
        gamma_phi = 0.0
        kappa = 0.0014322
        kappa_phi = 0.0
        n_c = 0.0283
        n_t = 0.0
        fd = 9.27594
        t_levels = 3
        c_levels = 3
        Parameters.__init__(self, fc=None, Ej=None, g=None, Ec=None, eps=None, fd=None, kappa=None, gamma=None, t_levels=None, c_levels=None, gamma_phi=None, kappa_phi=None, n_t=None,
                            n_c=None)