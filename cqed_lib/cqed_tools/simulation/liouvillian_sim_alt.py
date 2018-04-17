from .legion_tools import *
import scipy.sparse.linalg as lin

def liouvillian_sim_alt(job_index, output_directory='./results', eigenvalue=None, eigenstate=None):

    default_eigenvalue = 0

    if eigenvalue is None:
        eigenvalue = default_eigenvalue

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    stack_directory = output_directory

    kappa_phi = 0.0

    sys_params = stack_frame.iloc[job_index]
    frame_params = sys_params
    packaged_params = Parameters(frame_params.fc, frame_params.Ej, frame_params.g, frame_params.Ec, frame_params.eps,
                                 frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                 frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                 frame_params.n_c)

    print(stack_directory)

    directory = stack_directory + '/' + sys_params.group_folder + '/' + str(sys_params.job_index)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    print(directory)
    sys_params.to_csv('settings.csv')

    H = hamiltonian(packaged_params)
    c_ops = collapse_operators(packaged_params)

    L = liouvillian(H, c_ops)

    data = L.data
    csc = data.tocsc()

    if eigenstate is not None:
        if csc.shape[0] != eigenstate.shape[0]:
            eigenstate = None
            eigenvalue = default_eigenvalue

    values, states = lin.eigs(csc, k=10, sigma=eigenvalue, v0=eigenstate)
    sort_indices = np.argsort(np.abs(values))
    values = values[sort_indices]
    states = states[:,sort_indices]
    mi = pd.MultiIndex.from_tuples((frame_params.values[:],), names=frame_params.index)
    values = pd.DataFrame(values)
    values.columns = ['eigenvalues']
    states = pd.DataFrame(states)
    values.to_csv('eigenvalues.csv',index=False)

    values.index = mi
    os.chdir(stack_directory)
    values.to_hdf('results.h5',key='eigenvalues',append=True,format='table',mode='a')


    attempts = 0
    written = False
    while not written and attempts < 3:
        try:
            states.iloc[:,0:3].to_hdf('states.h5','states',mode='w')
            trial_opening = pd.read_hdf('states.h5')
            written = True
        except:
            attempts += 1
            print('failed to open')

    if not written:
        states.iloc[:,0:3].to_csv('states.csv')

    mask = np.abs(values) > 1e-10
    mask = mask.values[:,0]
    pruned_values = values.iloc[mask]
    chosen_index = pruned_values.index[np.argmin(np.abs(pruned_values).values)]

    os.chdir(cwd)

    return values.values[chosen_index,0], states.values[:,chosen_index]
