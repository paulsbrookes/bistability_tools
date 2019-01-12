import scipy.integrate
from scipy.special import factorial
import qutip.settings as qset
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip import *
import os
import types
import scipy.integrate
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from .gsl import *
from .hamiltonian_gen import *

from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.superoperator import spre, spost, liouvillian, mat2vec, vec2mat
from qutip.expect import expect_rho_vec
from qutip.solver import Options, Result, config
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.cy.codegen import Codegen
from qutip.cy.utilities import _cython_build_cleanup
from qutip.states import ket2dm
from qutip.rhs_generate import _td_format_check, _td_wrap_array_str
from qutip.settings import debug
from mpi4py import MPI


def str2bool(v):
    return v.lower() in ('yes', 'true', 'True', 't', 'T', '1', 'Y', 'y')


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
        # we have collapse operators, or rho0 is not a ket,
        # or H is a Liouvillian
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

        elif isinstance(H, list):
            # determine if we are dealing with list of [Qobj, string] or
            # [Qobj, function] style time-dependencies (for pure python and
            # cython, respectively)
            if n_func > 0:
                res = _mesolve_list_func_td_checkpoint(H, rho0, tlist, c_ops, e_ops, args, options, progress_bar, save,
                                                       subdir)
            else:
                res = _mesolve_list_str_td_checkpoint(H, rho0, tlist, c_ops, e_ops, args, options, progress_bar, save,
                                                      subdir)

        else:
            raise TypeError("Incorrect specification of Hamiltonian " +
                            "or collapse operators.")

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
    return _generic_ode_solve_checkpoint(r, rho0, tlist, e_ops, opt, progress_bar, save, subdir)


# -----------------------------------------------------------------------------
# A time-dependent dissipative master equation on the list-string format for
# cython compilation
#
def _mesolve_list_str_td_checkpoint(H_list, rho0, tlist, c_list, e_ops, args, opt,
                                    progress_bar, save, subdir):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state: must be a density matrix
    #
    if isket(rho0):
        rho0 = rho0 * rho0.dag()

    #
    # construct liouvillian
    #
    Lconst = 0
    Ldata = []
    Linds = []
    Lptrs = []
    Lcoeff = []
    Lobj = []

    # loop over all hamiltonian terms, convert to superoperator form and
    # add the data of sparse matrix representation to
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec

            if isoper(h):
                Lconst += -1j * (spre(h) - spost(h))
            elif issuper(h):
                Lconst += h
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Hamiltonian (expected operator or " +
                                "superoperator)")

        elif isinstance(h_spec, list):
            h = h_spec[0]
            h_coeff = h_spec[1]

            if isoper(h):
                L = -1j * (spre(h) - spost(h))
            elif issuper(h):
                L = h
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Hamiltonian (expected operator or " +
                                "superoperator)")

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            if isinstance(h_coeff, Cubic_Spline):
                Lobj.append(h_coeff.coeffs)
            Lcoeff.append(h_coeff)

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected string format)")

    # loop over all collapse operators
    for c_spec in c_list:

        if isinstance(c_spec, Qobj):
            c = c_spec

            if isoper(c):
                cdc = c.dag() * c
                Lconst += spre(c) * spost(c.dag()) - 0.5 * spre(cdc) \
                          - 0.5 * spost(cdc)
            elif issuper(c):
                Lconst += c
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Liouvillian (expected operator or " +
                                "superoperator)")

        elif isinstance(c_spec, list):
            c = c_spec[0]
            c_coeff = c_spec[1]

            if isoper(c):
                cdc = c.dag() * c
                L = spre(c) * spost(c.dag()) - 0.5 * spre(cdc) \
                    - 0.5 * spost(cdc)
                c_coeff = "(" + c_coeff + ")**2"
            elif issuper(c):
                L = c
            else:
                raise TypeError("Incorrect specification of time-dependent " +
                                "Liouvillian (expected operator or " +
                                "superoperator)")

            Ldata.append(L.data.data)
            Linds.append(L.data.indices)
            Lptrs.append(L.data.indptr)
            Lcoeff.append(c_coeff)

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected string format)")

    # add the constant part of the lagrangian
    if Lconst != 0:
        Ldata.append(Lconst.data.data)
        Linds.append(Lconst.data.indices)
        Lptrs.append(Lconst.data.indptr)
        Lcoeff.append("1.0")

    # the total number of liouvillian terms (hamiltonian terms +
    # collapse operators)
    n_L_terms = len(Ldata)

    # Check which components should use OPENMP
    omp_components = None
    if qset.has_openmp:
        if opt.use_openmp:
            omp_components = openmp_components(Lptrs)

    #
    # setup ode args string: we expand the list Ldata, Linds and Lptrs into
    # and explicit list of parameters
    #
    string_list = []
    for k in range(n_L_terms):
        string_list.append("Ldata[%d], Linds[%d], Lptrs[%d]" % (k, k, k))
    # Add object terms to end of ode args string
    for k in range(len(Lobj)):
        string_list.append("Lobj[%d]" % k)
    for name, value in args.items():
        if isinstance(value, np.ndarray):
            string_list.append(name)
        else:
            string_list.append(str(value))
    parameter_string = ",".join(string_list)

    #
    # generate and compile new cython code if necessary
    #
    if not opt.rhs_reuse or config.tdfunc is None:
        if opt.rhs_filename is None:
            config.tdname = "rhs" + str(os.getpid()) + str(config.cgen_num)
        else:
            config.tdname = opt.rhs_filename
        cgen = Codegen(h_terms=n_L_terms, h_tdterms=Lcoeff, args=args,
                       config=config, use_openmp=opt.use_openmp,
                       omp_components=omp_components,
                       omp_threads=opt.openmp_threads)
        cgen.generate(config.tdname + ".pyx")

        code = compile('from ' + config.tdname + ' import cy_td_ode_rhs',
                       '<string>', 'exec')
        exec (code, globals())
        config.tdfunc = cy_td_ode_rhs

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full()).ravel('F')
    if issuper(rho0):
        r = scipy.integrate.ode(_td_ode_rhs_super)
        code = compile('r.set_f_params([' + parameter_string + '])',
                       '<string>', 'exec')
    else:
        r = scipy.integrate.ode(config.tdfunc)
        code = compile('r.set_f_params(' + parameter_string + ')',
                       '<string>', 'exec')
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    exec (code, locals(), args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve_checkpoint(r, rho0, tlist, e_ops, opt, progress_bar, save, subdir)


# -----------------------------------------------------------------------------
# A time-dependent dissipative master equation on the list-function format
#
def _mesolve_list_func_td_checkpoint(H_list, rho0, tlist, c_list, e_ops, args, opt,
                                     progress_bar, save, subdir):
    """
    Internal function for solving the master equation. See mesolve for usage.
    """

    if debug:
        print(inspect.stack()[0][3])

    #
    # check initial state
    #
    if isket(rho0):
        rho0 = rho0 * rho0.dag()

    #
    # construct liouvillian in list-function format
    #
    L_list = []
    if opt.rhs_with_state:
        constant_func = lambda x, y, z: 1.0
    else:
        constant_func = lambda x, y: 1.0

    # add all hamitonian terms to the lagrangian list
    for h_spec in H_list:

        if isinstance(h_spec, Qobj):
            h = h_spec
            h_coeff = constant_func

        elif isinstance(h_spec, list) and isinstance(h_spec[0], Qobj):
            h = h_spec[0]
            h_coeff = h_spec[1]

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected callback function)")

        if isoper(h):
            L_list.append([(-1j * (spre(h) - spost(h))).data, h_coeff, False])

        elif issuper(h):
            L_list.append([h.data, h_coeff, False])

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "Hamiltonian (expected operator or superoperator)")

    # add all collapse operators to the liouvillian list
    for c_spec in c_list:

        if isinstance(c_spec, Qobj):
            c = c_spec
            c_coeff = constant_func
            c_square = False

        elif isinstance(c_spec, list) and isinstance(c_spec[0], Qobj):
            c = c_spec[0]
            c_coeff = c_spec[1]
            c_square = True

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected callback function)")

        if isoper(c):
            L_list.append([liouvillian(None, [c], data_only=True),
                           c_coeff, c_square])

        elif issuper(c):
            L_list.append([c.data, c_coeff, c_square])

        else:
            raise TypeError("Incorrect specification of time-dependent " +
                            "collapse operators (expected operator or " +
                            "superoperator)")

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full()).ravel('F')
    if issuper(rho0):
        if opt.rhs_with_state:
            r = scipy.integrate.ode(dsuper_list_td_with_state)
        else:
            r = scipy.integrate.ode(dsuper_list_td)
    else:
        if opt.rhs_with_state:
            r = scipy.integrate.ode(drho_list_td_with_state)
        else:
            r = scipy.integrate.ode(drho_list_td)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])
    r.set_f_params(L_list, args)

    #
    # call generic ODE code
    #
    return _generic_ode_solve_checkpoint(r, rho0, tlist, e_ops, opt, progress_bar, save, subdir)


def drho_list_td(t, rho, L_list, args):
    L = L_list[0][0] * L_list[0][1](t, args)
    for n in range(1, len(L_list)):
        #
        # L_args[n][0] = the sparse data for a Qobj in super-operator form
        # L_args[n][1] = function callback giving the coefficient
        #
        if L_list[n][2]:
            L = L + L_list[n][0] * (L_list[n][1](t, args)) ** 2
        else:
            L = L + L_list[n][0] * L_list[n][1](t, args)

    return L * rho

def _generic_ode_solve_checkpoint(r, rho0, tlist, e_ops, opt, progress_bar, save, subdir):
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

    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)
        print(1.0 * t / end_time)

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

            if False:
                if t_idx % 20 == 0:
                    rho_c = rho_checkpoint.ptrace(0)
                    with open('./cavity_states.pkl', 'ab') as f:
                        pickle.dump(rho_c, f)

            #with open('./results.csv', 'a') as file:
            results.to_csv('./results.csv', mode='a', header=first_row, float_format='%.15f')

            qsave(rho_checkpoint, './state_checkpoint')

        save = True

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if not opt.rhs_reuse and config.tdname is not None:
        _cython_build_cleanup(config.tdname)

    return output


def collapse_operators(params):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(params.kappa*(params.n_c+1)) * a)
    c_ops.append(np.sqrt(params.kappa*params.n_c) * a.dag())
    c_ops.append(np.sqrt(params.gamma*(params.n_t+1)) * sm)
    c_ops.append(np.sqrt(params.gamma*params.n_t) * sm.dag())
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


def hamiltonian(params):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    transmon_hamiltonian = transmon_hamiltonian_gen(params)
    coupling_hamiltonian = coupling_hamiltonian_gen(params)
    H = (params.fc - params.fd) * a.dag() * a + transmon_hamiltonian + coupling_hamiltonian + params.eps * (a + a.dag())
    return H


def maximum_yn(i, j, data):
    if data[i, j] > data[i + 1, j] and data[i, j] > data[i - 1, j]:
        if data[i, j] > data[i, j + 1] and data[i, j] > data[i, j - 1]:
            if data[i, j] > data[i + 1, j + 1] and data[i, j] > data[i - 1, j - 1]:
                if data[i, j] > data[i - 1, j + 1] and data[i, j] > data[i + 1, j - 1]:
                    return True
    return False


def mid_calc(point1, point2):
    midpoint = 0.5 * (point1 + point2)
    return midpoint


def line_func(x_array, m, midpoint):
    y_array = midpoint[1] + (x_array - midpoint[0]) * m
    return y_array


def m_calc(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    m = (float(x1) - float(x2)) / (float(y2) - float(y1))
    return m

def j_thresh_calc(row_indices, column_indices, i_array):
    i1, i2 = row_indices
    j1, j2 = column_indices
    idx_point1 = np.array([i1, j1])
    idx_point2 = np.array([i2, j2])

    m_idx = m_calc(idx_point1, idx_point2)
    midpoint_idx = mid_calc(idx_point1, idx_point2)
    j_thresh_array = line_func(i_array, m_idx, midpoint_idx)

    j_thresh_array = np.round(j_thresh_array)
    j_thresh_array = np.array([int(j) for j in j_thresh_array])

    return j_thresh_array

def maximum_finder(data):
    n_rows = data.shape[0]
    n_columns = data.shape[1]
    row_indices = []
    column_indices = []
    for i in range(1, n_rows - 1):
        for j in range(1, n_columns - 1):
            if maximum_yn(i, j, data):
                row_indices.append(i)
                column_indices.append(j)
    row_indices = np.array(row_indices)
    column_indices = np.array(column_indices)
    return row_indices, column_indices


def line_func_2(x, y, a):
    return x[np.newaxis, :] + a[:, np.newaxis] * (y - x)[np.newaxis, :]

def window_maximum_finder(i_limits, j_limits, array):
    array_reduced = array[i_limits[0]:i_limits[1], j_limits[0]:j_limits[1]]
    row_indices, column_indices = maximum_finder(array_reduced)
    if row_indices.shape[0] != 0 and column_indices.shape[0] != 0:
        array_values = array_reduced[row_indices, column_indices]
        max_index = np.argmax(array_values)
        return np.array([row_indices[max_index] + i_limits[0], column_indices[max_index] + j_limits[0]])
    else:
        return np.array([])


def bistable_states_calc(rho_ss, show=False, g=np.sqrt(2), axes=None):

    c_levels = rho_ss.dims[0][0]
    t_levels = rho_ss.dims[0][1]

    bistability = False
    rho_dim = None
    rho_bright = None
    characteristics = dict()

    n_bins = 101
    bistability_threshold = 1e-15
    rho_c = rho_ss.ptrace(0)
    xvec = np.linspace(-10, 10, n_bins)
    W = wigner(rho_c, xvec, xvec, g=g)
    W /= np.sum(W)

    if show:
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        cont0 = axes.contourf(xvec, xvec, W, 100)
        axes.plot([0, 0], axes.get_ylim())
        axes.plot(axes.get_xlim(), [0, 0])
        lbl2 = axes.set_title("Wigner")

    max_peak = window_maximum_finder([0, n_bins], [0, n_bins], W)

    if max_peak[1] > n_bins // 2:
        peak_bright = max_peak
        i_d_min = n_bins // 4
        i_d_max = n_bins * 3 // 4
        j_d_min = 0
        j_d_max = n_bins // 2 + (peak_bright[1] - (n_bins // 2)) // 2
        i_d_limits = [i_d_min, i_d_max]
        j_d_limits = [j_d_min, j_d_max]
        peak_dim = window_maximum_finder(i_d_limits, j_d_limits, W)
    else:
        peak_dim = max_peak
        i_b_min = 0
        i_b_max = n_bins // 2
        j_b_min = n_bins // 2
        j_b_max = n_bins
        i_b_limits = [i_b_min, i_b_max]
        j_b_limits = [j_b_min, j_b_max]
        peak_bright = window_maximum_finder(i_b_limits, j_b_limits, W)

    if peak_dim.shape[0] == 2 and peak_bright.shape[0] == 2:

        a_linspace = np.linspace(0, 1, n_bins)
        points = line_func_2(peak_dim, peak_bright, a_linspace)
        points = points.astype(int)
        W_values = []
        for point in points:
            W_values.append(W[point[0], point[1]])
        W_values = np.array(W_values)
        min_index = np.argmin(W_values)
        min_point = points[min_index]

        m = m_calc(peak_dim, peak_bright)
        i_array = np.arange(n_bins)
        threshold_array = line_func(i_array, m, min_point).astype(int)
        mask = (threshold_array >= 0) * (threshold_array < n_bins)
        j_array = threshold_array[mask]
        i_array = i_array[mask]

        mask_above = threshold_array >= n_bins
        mask_below = threshold_array < 0
        mask_within = (threshold_array >= 0) * (threshold_array < n_bins)
        threshold_array_sat = mask_within * threshold_array + mask_above * (n_bins - 1)

        if show:
            axes.scatter(xvec[peak_dim[1]], xvec[peak_dim[0]], color='r')
            axes.scatter(xvec[peak_bright[1]], xvec[peak_bright[0]], color='r')
            axes.scatter(xvec[j_array], xvec[i_array])

        p_bright = 0
        p_dim = 0
        for i, j in zip(range(n_bins), threshold_array_sat):
            p_bright += np.sum(W[i, j:])
            p_dim += np.sum(W[i, 0:j])

        contrast = 1 - np.abs(p_dim - p_bright)

        characteristics['p_dim'] = p_dim
        characteristics['p_bright'] = p_bright
        characteristics['contrast'] = contrast

        if contrast > bistability_threshold:
            bistability = True

            bright_alpha = np.sum(xvec[peak_bright] * np.array([1j, 1]))
            bright_projector = tensor(coherent_dm(c_levels, bright_alpha), qeye(t_levels))
            rho_bright = bright_projector * rho_ss
            rho_bright /= rho_bright.norm()

            dim_alpha = np.sum(xvec[peak_dim] * np.array([1j, 1]))
            dim_projector = tensor(coherent_dm(c_levels, dim_alpha), qeye(t_levels))
            rho_dim = dim_projector * rho_ss
            rho_dim /= rho_dim.norm()

            characteristics['alpha_bright'] = bright_alpha
            characteristics['alpha_dim'] = dim_alpha

    if show:
        plt.show()

    return bistability, rho_dim, rho_bright, characteristics

def manage_jobs():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    n_active = size

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)
        stack_frame = stack_frame.set_index('job_index')

    n_jobs = stack_frame.shape[0]

    if os.path.exists('register.csv'):
        with open('register.csv', 'r') as f:
            register = pd.read_csv(f)
        register.running = np.zeros(register.shape[0], dtype=int)
        register = register.set_index('job_index')
        with open('register.csv', 'w') as f:
            register.to_csv(f)
    else:
        register = np.zeros([stack_frame.shape[0], 2], dtype=int)
        register = pd.DataFrame(register)
        register.columns = ['completed', 'running']
        register.index.name = 'job_index'
        with open('register.csv', 'w') as f:
            register.to_csv(f)

    #for job_index in range(n_jobs):
    #    stack_frame.iloc[job_index, -1] = 0
    #with open('stack.csv', 'w') as f:
    #    f.write(stack_name + '\n')
    #    stack_frame.to_csv(f)

    while True:

        message = np.empty(1, dtype='i')
        comm.Recv(message, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == 1:
            print('complete message received and recognised')
            completed_job_index = message[0]
            register.iloc[completed_job_index,0] = 1
            register.iloc[completed_job_index,1] = 0
            with open('register.csv', 'w') as f:
                register.to_csv(f)

        allocated_job = False
        job_index = 0

        while not allocated_job:
            register_row = register.iloc[job_index]
            if not register_row.completed:
                if not register_row.running:
                    packaged_index = np.array([job_index], dtype='i')
                    comm.Send(packaged_index, dest=source, tag=3)
                    allocated_job = True
                    register.iloc[job_index, 1] = 1
                    with open('register.csv', 'w') as f:
                        register.to_csv(f)
            job_index += 1
            if job_index == n_jobs and not allocated_job:
                allocated_job = True
                message = np.empty(1, dtype='i')
                comm.Send(message, dest=source, tag=4)
                n_active -= 1
                if n_active == 1:
                    print('main thread shutdown')
                    exit()


def mpi_allocator(job, args, kwargs):
    print('In mpi_allocator we have kwargs = ',kwargs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()
    if rank == 0:
        manage_jobs()
        exit()

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    home = os.environ['HOME']

    completed = False

    while True:

        if completed:
            comm.Send(np.array([job_index]), dest=0, tag=1)
            print('complete message sent')
            completed = False
        else:
            message = np.empty(1, dtype='i')
            comm.Send(message, dest=0, tag=0)
        packaged_index = np.empty(1, dtype='i')
        comm.Recv(packaged_index, source=0, status=status)
        tag = status.Get_tag()
        if tag == 4:
            exit()
        job_index = packaged_index[0]

        job(job_index, *args, **kwargs)

        completed = True


def eigenstate_check_base(op, state_psi):
    state_chi = op * state_psi
    norm = state_chi.norm()
    if norm == 0.0:
        metric = 1.0
    else:
        metric = state_psi.dag() * state_chi / norm
        metric = np.abs(metric[0, 0])
    return metric


eigenstate_check = np.vectorize(eigenstate_check_base)