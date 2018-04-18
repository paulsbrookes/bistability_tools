from .legion_tools import *
import scipy.sparse.linalg as lin
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode

def hdf_append(path,data,key):
    if os.path.exists(path):
        loaded = pd.read_hdf(path,key=key)
    else:
        loaded= pd.DataFrame()
    combined = loaded.append(data)
    combined.to_hdf(path,key=key,mode='a')

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

    k = 10
    values, states = lin.eigs(csc, k=k, sigma=eigenvalue, v0=eigenstate)
    sort_indices = np.argsort(np.abs(values))
    values = values[sort_indices]
    states = states[:,sort_indices]
    #mi = pd.MultiIndex.from_tuples((frame_params.values[:],), names=frame_params.index)

    values = pd.DataFrame(values)
    values.columns = ['eigenvalues']
    states = pd.DataFrame(states)
    values.to_csv('eigenvalues.csv',index=False)


    tuples = []
    arrays = []
    for i in range(k):
        indices = list(frame_params.values)
        indices.append(i)
        tuples.append(tuple(indices))
        arrays.append(indices)
    names = list(frame_params.index)
    names.append('index')
    mi = pd.MultiIndex.from_tuples(tuples, names=names)
    #values.index = mi
    saving = values.copy()
    saving.index = mi
    os.chdir(stack_directory)
    print('saving results.h5')
    if os.path.exists('results.h5'):
        loaded = pd.read_hdf('results.h5',key='eigenvalues')
    else:
        loaded = pd.DataFrame()
    combined = loaded.append(saving)
    combined.to_hdf('results.h5',key='eigenvalues',mode='a')
    #values.to_hdf('results.h5',key='eigenvalues',append=True,format='table',mode='a')

    n = packaged_params.t_levels * packaged_params.c_levels
    dims = [packaged_params.c_levels, packaged_params.t_levels]
    ground_state_vector = states.values[:, 0]
    data = dense2D_to_fastcsr_fmode(np.asfortranarray(vec2mat(ground_state_vector)), n, n)
    rho = Qobj(data, dims=[dims, dims], isherm=True)
    rho = rho + rho.dag()
    rho /= rho.tr()

    a = tensor(destroy(packaged_params.c_levels), qeye(packaged_params.t_levels))

    a_exp_point = expect(a, rho)
    n_exp_point = expect(a.dag() * a, rho)
    a_exp_point = pd.DataFrame([a_exp_point], index=mi[0:1])
    n_exp_point = pd.DataFrame([n_exp_point], index=mi[0:1])

    hdf_append('results.h5', a_exp_point, 'a')
    hdf_append('results.h5', n_exp_point, 'n')





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


