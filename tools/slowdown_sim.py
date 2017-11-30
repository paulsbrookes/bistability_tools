from legion_tools import *


def slowdown_sim(job_index, output_directory='./results'):

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    stack_directory = output_directory + '/' + stack_name

    kappa_phi = 0.0

    sys_params = stack_frame.iloc[job_index]
    frame_params = sys_params
    packaged_params = Parameters(frame_params.fc, frame_params.Ej, frame_params.g, frame_params.Ec, frame_params.eps,
                                 frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                 frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                 frame_params.n_c)
    directory = stack_directory + '/' + sys_params.group_folder + '/' + str(job_index)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    print(directory)
    sys_params.to_csv('settings.csv')

    H = hamiltonian(packaged_params)
    c_ops = collapse_operators(packaged_params)
    options = Options(nsteps=200000)

    if os.path.exists('./steady_state.qu'):
        if os.path.exists('./state_checkpoint.qu'):
            print('Loading state checkpoint for job_index = '+str(job_index))
            initial_state = qload('./state_checkpoint')
            previous_results = pd.read_csv('./results.csv')
            delta_t = 1.0 * sys_params.end_time / (sys_params.snapshots - 1)
            start_time = float(previous_results['times'].iloc[-1])
            new_snapshots = sys_params.snapshots - start_time / delta_t
            snapshot_times = np.linspace(start_time, sys_params.end_time, new_snapshots)
            save = False #don't save the first row of results, it's already there
            bistability = True
        else:
            rho_ss = qload('steady_state')
            bistability, rho_dim, rho_bright, characteristics = bistable_states_calc(rho_ss)
            if sys_params.qubit_state == 0:
                initial_state = rho_dim
            else:
                initial_state = rho_bright
            bistability_characteristics = [bistability, rho_dim, rho_bright, characteristics]
            qsave(bistability_characteristics, './characteristics')
            start_time = 0
            snapshot_times = np.linspace(start_time, sys_params.end_time, sys_params.snapshots)
            save = True #save the first row of results

    else:
        print('Finding steady state for job_index = '+str(job_index))
        rho_ss = steadystate(H, c_ops)
        qsave(rho_ss, './steady_state')
        bistability, rho_dim, rho_bright, characteristics = bistable_states_calc(rho_ss)
        bistability_characteristics = [bistability, rho_dim, rho_bright, characteristics]
        qsave(bistability_characteristics, './characteristics')
        if sys_params.qubit_state == 0:
            initial_state = rho_dim
        else:
            initial_state = rho_bright
        start_time = 0
        snapshot_times = np.linspace(start_time, sys_params.end_time, sys_params.snapshots)
        save = True

    a = tensor(destroy(sys_params.c_levels), qeye(sys_params.t_levels))
    sm = tensor(qeye(sys_params.c_levels), destroy(sys_params.t_levels))

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

    if bistability:
        output = mesolve_checkpoint(H, initial_state, snapshot_times, c_ops, e_ops, save, directory, options=options)

    os.chdir(cwd)
