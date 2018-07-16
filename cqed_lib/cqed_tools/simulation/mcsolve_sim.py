from .legion_tools import *
from .hamiltonian import *


def mcsolve_sim(job_index, stack_directory='./results', transmon=True):

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
    sys_params.to_csv('settings.csv')

    options = Options(nsteps=2000000000)

    start_time = 0
    snapshot_times = np.linspace(start_time, sys_params.end_time, sys_params.snapshots)

    a = tensor(destroy(sys_params.c_levels), qeye(sys_params.t_levels))
    sm = tensor(qeye(sys_params.c_levels), destroy(sys_params.t_levels))
    initial_state = tensor(basis(sys_params.c_levels, 0), basis(sys_params.t_levels, 0))

    options.store_final_state = True
    options.ntraj = sys_params.ntraj
    num_cpus_max = 4
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

    results = mcsolve(H, initial_state, snapshot_times, c_ops, e_ops, map_func=map_func, options=options)
    qsave(results, 'results')

    results_dict = OrderedDict()
    for key in e_ops.keys():
        results_dict[key] = results.expect[key]
    results_csv = pd.DataFrame(results_dict, index=snapshot_times)
    results_csv.index.name = 'times'
    results_csv.to_csv('results.csv')

    os.chdir(cwd)