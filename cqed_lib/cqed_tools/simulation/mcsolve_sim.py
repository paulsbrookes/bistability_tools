from .legion_tools import *
from .hamiltonian_gen import *


def mcsolve_sim(job_index, stack_directory='./results', transmon=True, bistable_initial=True):

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    kappa_phi = 0.0

    sys_params = stack_frame.iloc[job_index]
    frame_params = sys_params


    if transmon is True:
        packaged_params = Parameters(fc=frame_params.fc, Ej=frame_params.Ej, g=frame_params.g, Ec=frame_params.Ec, eps=frame_params.eps,
                                     fd=frame_params.fd, kappa=frame_params.kappa, gamma=frame_params.gamma, t_levels=frame_params.t_levels,
                                     c_levels=frame_params.c_levels, gamma_phi=frame_params.gamma_phi, kappa_phi=kappa_phi, n_t=frame_params.n_t,
                                     n_c=frame_params.n_c, ntraj=frame_params.ntraj)
        H = hamiltonian(packaged_params, transmon=transmon)
        qsave(H, 'slowdown_hamiltonian')
        c_ops = collapse_operators(packaged_params)
    else:
        packaged_params = Parameters(fc=frame_params.fc, Ej=None, g=frame_params.g, Ec=None, eps=frame_params.eps,
                                     fd=frame_params.fd, kappa=frame_params.kappa, gamma=frame_params.gamma, t_levels=frame_params.t_levels,
                                     c_levels=frame_params.c_levels, gamma_phi=frame_params.gamma_phi, kappa_phi=kappa_phi, n_t=frame_params.n_t,
                                     n_c=frame_params.n_c, f01=frame_params.f01, ntraj=frame_params.ntraj)
        H = hamiltonian(packaged_params, transmon=transmon)
        qsave(H, 'slowdown_hamiltonian')
        c_ops = collapse_operators(packaged_params)


    directory = stack_directory + '/' + sys_params.group_folder + '/' + str(sys_params.job_index)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    print(directory)
    sys_params.to_csv('settings.csv', header=False)

    #options = Options(nsteps=2000000000)

    start_time = 0
    snapshot_times = np.linspace(start_time, sys_params.end_time, sys_params.snapshots)

    a = tensor(destroy(sys_params.c_levels), qeye(sys_params.t_levels))
    sm = tensor(qeye(sys_params.c_levels), destroy(sys_params.t_levels))

    if bistable_initial:
        if os.path.exists('./steady_state.qu'):
            rho_ss = qload('steady_state')
        else:
            print('Finding steady state for job_index = ' + str(sys_params.job_index))
            rho_ss = steadystate(H, c_ops)
            qsave(rho_ss, './steady_state')
        bistability, rho_dim, rho_bright, characteristics = bistable_states_calc(rho_ss)
        bistability_characteristics = [bistability, rho_dim, rho_bright, characteristics]
        qsave(bistability_characteristics, './characteristics')
        if sys_params.qubit_state == 0:
            chosen_state = rho_dim
        else:
            chosen_state = rho_bright
        sm_expect = expect(sm, chosen_state)
        a_expect = expect(a, chosen_state)
        initial_state = tensor(coherent(sys_params.c_levels, a_expect), coherent(sys_params.t_levels, sm_expect))
    else:
        initial_state = tensor(basis(sys_params.c_levels, 0), basis(sys_params.t_levels, 0))


    options = Options()
    options.store_final_state = True
    options.ntraj = sys_params.ntraj
    num_cpus_max = 8
    options.num_cpus = min(num_cpus_max, options.ntraj)
    if options.num_cpus > 1:
        map_func = parallel_map
        print('using parallel_map')
    else:
        map_func = serial_map
        print('using serial_map')

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

    print('Sending job_index = ' + str(job_index) + ' to mcsolve.')
    print(map_func,options.ntraj,snapshot_times.shape,snapshot_times[-1])
    results = mcsolve(H, initial_state, snapshot_times, c_ops, e_ops, map_func=map_func, options=options)
    print('Finished mcsolve for job_Index = ' + str(job_index) + ' to mcsolve.')
    qsave(results, 'results')

    results_dict = OrderedDict()
    for key in e_ops.keys():
        results_dict[key] = results.expect[key]
    results_csv = pd.DataFrame(results_dict, index=snapshot_times)
    results_csv.index.name = 'times'
    results_csv.to_csv('results.csv')

    os.chdir(cwd)
