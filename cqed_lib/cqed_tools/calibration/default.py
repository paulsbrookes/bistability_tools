from .spectra_tools import Parameters

class DefaultParameters(Parameters):
    def __init__(self):
        Ec = 0.21448180630636157
        fc = 10.4263
        Ej = 52.405522658165175
        g = 0.29324769334758782
        eps = 0.0
        gamma = 0.0
        gamma_phi = 0.0
        kappa = 0.0008
        kappa_phi = 0.0
        n_c = 0.037
        n_t = 0.0
        fd = 9.27594
        t_levels = 3
        c_levels = 3
        Parameters.__init__(self, fc, Ej, g, Ec, eps, fd, kappa, gamma, t_levels, c_levels, gamma_phi, kappa_phi, n_t,
                            n_c)