import numpy as np
from qutip import *
from datetime import datetime
import os
import shutil
import json
import argparse
import types
from scipy.special import factorial
from tqdm import tqdm
from functools import partial
import numpy as np
import scipy.sparse as sp
import scipy.integrate
import warnings
import pandas as pd
import subprocess
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from collections import OrderedDict

from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.superoperator import spre, spost, liouvillian, mat2vec, vec2mat
from qutip.expect import expect_rho_vec
from qutip.solver import Options, Result, config
from qutip.cy.spmatfuncs import cy_ode_rhs, cy_ode_rho_func_td
from qutip.cy.codegen import Codegen
from qutip.cy.utilities import _cython_build_cleanup
from qutip.rhs_generate import rhs_generate
from qutip.states import ket2dm
from qutip.rhs_generate import _td_format_check, _td_wrap_array_str
from qutip.settings import debug

from qutip.sesolve import (_sesolve_list_func_td, _sesolve_list_str_td,
                           _sesolve_list_td, _sesolve_func_td, _sesolve_const)

from qutip.ui.progressbar import BaseProgressBar, TextProgressBar

class TransientOptions:
    def __init__(self, endtime=100000, snapshots=201, display=False):
        self.display = display
        self.endtime = endtime
        self.snapshots = snapshots

def decay_model(t, lam, c):
    occupation = np.exp(lam*t + c)
    return occupation

def decay_func(t, A, T, B):
    y = A*np.exp(-t/T) + B
    return y

def decay_fit(t, y, p0=[0.5, 2.0, 0.04]):
    popt, pcov = curve_fit(decay_func, t, y, p0=p0)
    return popt, pcov

def decaying_wave(t, A, omega, phi, T):
    signal = A*np.cos(omega*t + phi)*np.exp(-t/T)
    return signal

def mesolve_checkpoint(H, rho0, tlist, c_ops, e_ops, save, subdir, args={}, options=None,
            progress_bar=None):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian (`H`) and an [optional] set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system. In the absence of collapse operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows to solve master equations that are not on standard Lindblad
    form by passing a custom Liouvillian in place of either the `H` or `c_ops`
    elements.

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be callback
    functions that takes two arguments, time and `args`, and returns the
    Hamiltonian or Liouvillian for the system at that point in time
    (*callback format*).

    Alternatively, `H` and `c_ops` can be a specified in a nested-list format
    where each element in the list is a list of length 2, containing an
    operator (:class:`qutip.qobj`) at the first element and where the
    second element is either a string (*list string format*), a callback
    function (*list callback format*) that evaluates to the time-dependent
    coefficient for the corresponding operator, or a NumPy array (*list
    array format*) which specifies the value of the coefficient to the
    corresponding operator for each value of t in tlist.

    *Examples*

        H = [[H0, 'sin(w*t)'], [H1, 'sin(2*w*t)']]

        H = [[H0, f0_t], [H1, f1_t]]

        where f0_t and f1_t are python functions with signature f_t(t, args).

        H = [[H0, np.sin(w*tlist)], [H1, np.sin(2*w*tlist)]]

    In the *list string format* and *list callback format*, the string
    expression and the callback function must evaluate to a real or complex
    number (coefficient for the corresponding operator).

    In all cases of time-dependent operators, `args` is a dictionary of
    parameters that is used when evaluating operators. It is passed to the
    callback functions as second argument.

    **Additional options**

    Additional options to mesolve can be set via the `options` argument, which
    should be an instance of :class:`qutip.solver.Options`. Many ODE
    integration options can be set this way, and the `store_states` and
    `store_final_state` options can be used to store states even though
    expectation values are requested via the `e_ops` argument.

    .. note::

        If an element in the list-specification of the Hamiltonian or
        the list of collapse operators are in superoperator form it will be
        added to the total Liouvillian of the problem with out further
        transformation. This allows for using mesolve for solving master
        equations that are not on standard Lindblad form.

    .. note::

        On using callback function: mesolve transforms all :class:`qutip.qobj`
        objects to sparse matrices before handing the problem to the integrator
        function. In order for your callback function to work correctly, pass
        all :class:`qutip.qobj` objects that are used in constructing the
        Hamiltonian via args. mesolve will check for :class:`qutip.qobj` in
        `args` and handle the conversion to sparse matrices. All other
        :class:`qutip.qobj` objects that are not passed via `args` will be
        passed on to the integrator in scipy which will raise an NotImplemented
        exception.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian, or a callback function for time-dependent
        Hamiltonians, or alternatively a system Liouvillian.

    rho0 : :class:`qutip.Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of :class:`qutip.Qobj`
        single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : :class:`qutip.Options`
        with options for the solver.

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------

    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        either an *array* `result.expect` of expectation values for the times
        specified by `tlist`, or an *array* `result.states` of state vectors or
        density matrices corresponding to the times in `tlist` [if `e_ops` is
        an empty list], or nothing if a callback function was given in place of
        operators for which to calculate the expectation values.

    """

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    # check whether c_ops or e_ops is is a single operator
    # if so convert it to a list containing only that operator
    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    # convert array based time-dependence to string format
    H, c_ops, args = _td_wrap_array_str(H, c_ops, args, tlist)

    # check for type (if any) of time-dependent inputs
    _, n_func, n_str = _td_format_check(H, c_ops)

    if options is None:
        options = Options()

    if (not options.rhs_reuse) or (not config.tdfunc):
        # reset config collapse and time-dependence flags to default values
        config.reset()

    res = None

    #
    # dispatch the appropriate solver
    #
    if ((c_ops and len(c_ops) > 0)
        or (not isket(rho0))
        or (isinstance(H, Qobj) and issuper(H))
        or (isinstance(H, list) and
            isinstance(H[0], Qobj) and issuper(H[0]))):

        #
        # we have collapse operators
        #

        #
        # find out if we are dealing with all-constant hamiltonian and
        # collapse operators or if we have at least one time-dependent
        # operator. Then delegate to appropriate solver...
        #

        if isinstance(H, Qobj):
            # constant hamiltonian
            if n_func == 0 and n_str == 0:
                # constant collapse operators
                res = mesolve_const_checkpoint(H, rho0, tlist, c_ops,
                                     e_ops, args, options,
                                     progress_bar, save, subdir)

    res.expect = {e: res.expect[n]
                    for n, e in enumerate(e_ops.keys())}

    return res


# -----------------------------------------------------------------------------
# Master equation solver
#
def mesolve_const_checkpoint(H, rho0, tlist, c_op_list, e_ops, args, opt,
                   progress_bar, save, subdir):
    """
    Evolve the density matrix using an ODE solver, for constant hamiltonian
    and collapse operators.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = ket2dm(rho0)

    #
    # construct liouvillian
    #
    if opt.tidy:
        H = H.tidyup(opt.atol)

    L = liouvillian(H, c_op_list)

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full()).ravel()
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(L.data.data, L.data.indices, L.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # call generic ODE code
    #
    return generic_ode_solve_checkpoint(r, rho0, tlist, e_ops, opt, progress_bar, save, subdir)

def generic_ode_solve_checkpoint(r, rho0, tlist, e_ops, opt, progress_bar, save, subdir):
    """
    Internal function for solving ME. Solve an ODE which solver parameters
    already setup (r). Calculate the required expectation values or invoke
    callback function at each time step.
    """

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    e_sops_data = []

    output = Result()
    output.solver = "mesolve"
    output.times = tlist

    if opt.store_states:
        output.states = []

    e_ops_dict = e_ops
    e_ops = [e for e in e_ops_dict.values()]
    headings = [key for key in e_ops_dict.keys()]

    if isinstance(e_ops, types.FunctionType):
        n_expt_op = 0
        expt_callback = True

    elif isinstance(e_ops, list):

        n_expt_op = len(e_ops)
        expt_callback = False

        if n_expt_op == 0:
            # fall back on storing states
            output.states = []
            opt.store_states = True
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in e_ops:
                e_sops_data.append(spre(op).data)
                if op.isherm and rho0.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    results_row = np.zeros(n_expt_op)

    #
    # start evolution
    #
    progress_bar.start(n_tsteps)

    rho = Qobj(rho0)
    dims = rho.dims

    dt = np.diff(tlist)

    end_time = tlist[-1]

    for t_idx, t in tqdm(enumerate(tlist)):
        progress_bar.update(t_idx)
        #print 1.0*t/end_time

        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")

        if opt.store_states or expt_callback:
            rho.data = vec2mat(r.y)

            if opt.store_states:
                output.states.append(Qobj(rho))

            if expt_callback:
                # use callback method
                e_ops(t, rho)

        for m in range(n_expt_op):
            if output.expect[m].dtype == complex:
                output.expect[m][t_idx] = expect_rho_vec(e_sops_data[m],
                                                         r.y, 0)
                results_row[m] = output.expect[m][t_idx]
            else:
                output.expect[m][t_idx] = expect_rho_vec(e_sops_data[m],
                                                         r.y, 1)
                results_row[m] = output.expect[m][t_idx]


        results = pd.DataFrame(results_row).T
        results.columns = headings
        results.index = [t]
        results.index.name = 'times'
        if t == 0:
            first_row = True
        else:
            first_row = False
        if save:

            rho_checkpoint = Qobj(vec2mat(r.y))
            rho_checkpoint.dims = dims

            if t_idx % 200 == 0:
                rho_c = rho_checkpoint.ptrace(0)
                with open('./cavity_states.pkl', 'ab') as f:
                    pickle.dump(rho_c, f)

            with open('./results.csv', 'a') as file:
                results.to_csv(file, header=first_row, float_format='%.15f')

            qsave(rho_checkpoint, './state_checkpoint')

        save = True

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if not opt.rhs_reuse and config.tdname is not None:
        _cython_build_cleanup(config.tdname)

    return output

class simulation_options:
    def __init__(self, end_time, n_snaps):
        self.end_time = end_time
        self.n_snaps = n_snaps