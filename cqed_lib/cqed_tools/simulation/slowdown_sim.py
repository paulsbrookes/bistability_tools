from .legion_tools import *
from .hamiltonian_gen import *
from ..mf import mf_calc


def slowdown_sim(job_index, output_directory='./results', bistable_initial=True, transmon=True, transformation=False, mf_init=True, g=np.sqrt(2)):

    bistable_initial = bistable_initial
    transmon = transmon

    print('In slowdown_sim.py we have bistable_initial = ' + str(bistable_initial) + ' for job_index = ' + str(job_index))

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    stack_directory = output_directory

    kappa_phi = 0.0

    sys_params = stack_frame.iloc[job_index]
    frame_params = sys_params

    if transmon is True:
        packaged_params = Parameters(frame_params.fc, frame_params.Ej, frame_params.g, frame_params.Ec,
                                     frame_params.eps,
                                     frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                     frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                     frame_params.n_c)
        H = hamiltonian(packaged_params, transmon=transmon)
        c_ops = collapse_operators(packaged_params)
    else:
        packaged_params = Parameters(frame_params.fc, None, frame_params.g, None, frame_params.eps,
                                     frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                     frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                     frame_params.n_c, frame_params.f01)
        H = hamiltonian(packaged_params, transmon=transmon)
        c_ops = collapse_operators(packaged_params)

    a = tensor(destroy(sys_params.c_levels), qeye(sys_params.t_levels))
    sm = tensor(qeye(sys_params.c_levels), destroy(sys_params.t_levels))

    directory = stack_directory + '/' + sys_params.group_folder + '/' + str(sys_params.job_index)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    print('The working directory for the current job index is ' + str(directory))
    sys_params.to_csv('settings.csv')

    options = Options(nsteps=2000000000)

    if os.path.exists('./state_checkpoint.qu'):
        print('Loading state checkpoint for job_index = '+str(sys_params.job_index))
        initial_state = qload('./state_checkpoint')
        H = qload('./hamiltonian')
        c_ops = qload('./c_ops')
        previous_results = pd.read_csv('./results.csv')
        delta_t = 1.0 * sys_params.end_time / (sys_params.snapshots - 1)
        start_time = float(previous_results['times'].iloc[-1])
        new_snapshots = sys_params.snapshots - start_time / delta_t
        snapshot_times = np.linspace(start_time, sys_params.end_time, new_snapshots)
        save = False #don't save the first row of results, it's already there
        bistability = True
    else:
        start_time = 0
        snapshot_times = np.linspace(start_time, sys_params.end_time, sys_params.snapshots)
        save = True #save the first row of results
        if bistable_initial is True:
            bistability_characteristics = dict()
            if os.path.exists('./steady_state.qu'):
                rho_ss = qload('steady_state')
                if mf_init:
                    mf_amplitudes = mf_calc(packaged_params)
                    if mf_amplitudes.dropna().shape[0] == 4:
                        bistability = True

                        bright_alpha = mf_amplitudes.a_bright
                        bright_projector = tensor(coherent_dm(packaged_params.c_levels, g*bright_alpha), qeye(packaged_params.t_levels))
                        rho_bright = bright_projector * rho_ss
                        rho_bright /= rho_bright.norm()

                        dim_alpha = mf_amplitudes.a_dim
                        dim_projector = tensor(coherent_dm(packaged_params.c_levels, g*dim_alpha), qeye(packaged_params.t_levels))
                        rho_dim = dim_projector * rho_ss
                        rho_dim /= rho_dim.norm()

                        characteristics = None

                    else:
                        bistability = False
                        rho_dim = None
                        rho_bright = None
                        characteristics = None
                else:
                    raise AssertionError
                    #bistability, rho_dim, rho_bright, characteristics = bistable_states_calc(rho_ss)
                if sys_params.qubit_state == 0:
                    print('Dim initial state.')
                    initial_state = rho_dim
                else:
                    print('Bright initial state.')
                    initial_state = rho_bright
            else:
                print('Finding steady state for job_index = '+str(sys_params.job_index))
                rho_ss = steadystate(H, c_ops)
                qsave(rho_ss, './steady_state')
                bistability, rho_dim, rho_bright, characteristics = bistable_states_calc(rho_ss)
                if sys_params.qubit_state == 0:
                    print('Dim initial state.')
                    initial_state = rho_dim
                else:
                    print('Bright initial state.')
                    initial_state = rho_bright
            if transformation and bistability:
                alpha_bright = expect(a,rho_bright)
                alpha_dim = expect(a,rho_dim)
                bistability_characteristics['alpha_bright'] = alpha_bright
                bistability_characteristics['alpha_dim'] = alpha_dim
                alpha = 0.5*(alpha_bright+alpha_dim)
                beta = 0.0
            else:
                alpha = 0.0
                beta = 0.0
            bistability_characteristics['bistability'] = bistability
            bistability_characteristics['rho_dim'] = rho_dim
            bistability_characteristics['rho_bright'] = rho_bright
            bistability_characteristics['characteristics'] = characteristics
            bistability_characteristics['alpha'] = alpha
            bistability_characteristics['beta'] = beta
            qsave(bistability_characteristics, './characteristics')
        else:
            print('Choosing initial state in the transmon basis.')
            initial_state = tensor(fock_dm(sys_params.c_levels,0), fock_dm(sys_params.t_levels, sys_params.qubit_state))
            bistability = None
            alpha = 0.0
            beta = 0.0

        if transmon is True:
            packaged_params = Parameters(frame_params.fc, frame_params.Ej, frame_params.g, frame_params.Ec,
                                         frame_params.eps,
                                         frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                         frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                         frame_params.n_c)
            H = hamiltonian(packaged_params, transmon=transmon, alpha=alpha, beta=beta)
            c_ops = collapse_operators(packaged_params, alpha=alpha, beta=beta)
        else:
            packaged_params = Parameters(frame_params.fc, None, frame_params.g, None, frame_params.eps,
                                         frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                         frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                         frame_params.n_c, frame_params.f01)
            H = hamiltonian(packaged_params, transmon=transmon, alpha=alpha, beta=beta)
            c_ops = collapse_operators(packaged_params, alpha=alpha, beta=beta)

        qsave(H, 'hamiltonian')
        qsave(c_ops, 'c_ops')

    options.store_final_state = True

    e_ops = OrderedDict()
    e_ops['a_op_re'] = (a + a.dag()) / 2
    e_ops['a_op_im'] = -1j * (a - a.dag()) / 2
    e_ops['photons'] = a.dag() * a
    for level in range(sys_params.c_levels):
        e_ops['c_level_' + str(level)] = tensor(fock_dm(sys_params.c_levels, level), qeye(sys_params.t_levels))
    e_ops['sm_op_re'] = (sm.dag() + sm) / 2
    e_ops['sm_op_im'] = -1j * (sm - sm.dag()) / 2
    e_ops['excitations'] = sm.dag() * sm
    for level in range(sys_params.t_levels):
        e_ops['t_level_' + str(level)] = tensor(qeye(sys_params.c_levels), fock_dm(sys_params.t_levels, level))

    qsave(H,'slowdown_hamiltonian')

    if bistability or not bistable_initial:
        print('hermitian',H.isherm)
        print('Going into the mesolve function we have a_op_re = ' + str(expect(e_ops['a_op_re'],initial_state)))
        output = mesolve_checkpoint(H, initial_state, snapshot_times, c_ops, e_ops, save, directory, options=options)

    os.chdir(cwd)
