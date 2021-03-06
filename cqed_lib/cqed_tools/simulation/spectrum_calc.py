from .legion_tools import *
from collections import OrderedDict


def spectrum_calc(job_index, output_directory='./results', save_state=False):


    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_frame = pd.read_csv(f)

    kappa_phi = 0.0

    sys_params = stack_frame.iloc[job_index]
    frame_params = sys_params
    packaged_params = Parameters(frame_params.fc, frame_params.Ej, frame_params.g, frame_params.Ec, frame_params.eps,
                                 frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                 frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                 frame_params.n_c)
    directory = output_directory + '/' + sys_params.group_folder + '/' + str(sys_params.job_index)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    print('Entering directory: '+directory)
    sys_params.to_csv('settings.csv', header=False)

    H = hamiltonian(packaged_params)
    qsave(H,'spectrum_hamiltonian')
    c_ops = collapse_operators(packaged_params)
    a = tensor(destroy(sys_params.c_levels), qeye(sys_params.t_levels))
    sm = tensor(qeye(sys_params.c_levels), destroy(sys_params.t_levels))

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

    if not os.path.exists('./steady_state.qu'):
        if save_state or not os.path.exists('./ss_results.csv'):
            print('Generating steady state for job_index = '+str(sys_params.job_index))
            try:
                rho_ss = steadystate(H, c_ops)
                if save_state:
                    qsave(rho_ss,'steady_state')

                expectations = []
                for e_op in e_ops.values():
                    expectations.append(expect(e_op, rho_ss))

                headings = [key for key in e_ops.keys()]
                results = pd.DataFrame(expectations).T
                results.columns = headings
                results.index = [sys_params.job_index]
                results.index.name = 'job_index'

                with open('./ss_results.csv', 'a') as file:
                    results.to_csv(file, float_format='%.15f')
            except:
                print('failure')

    os.chdir(cwd)
