from .legion_tools import *
from .hamiltonian import *
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
import scipy.sparse.linalg as lin
import h5py


def hdf_append(path,data,key):
    if os.path.exists(path):
        f = h5py.File(path, 'r')
        keys = f.keys()
        f.close()
    else:
        keys = []

    if key in keys:
        loaded = pd.read_hdf(path,key=key)
    else:
        loaded = pd.DataFrame()

    combined = loaded.append(data)
    combined.to_hdf(path,key=key,mode='a')


def eliminate(params):
    delta_a = params.fc - params.fd
    delta_b = params.f01 - params.fd
    delta_eff = delta_b - params.g**2 * delta_a/(delta_a**2 + params.kappa**2)
    kappa_eff = params.gamma + params.g**2 * params.kappa/(delta_a**2 + params.kappa**2)
    eps_1_eff = params.g*params.kappa*params.eps/(delta_a**2 + params.kappa**2)
    eps_2_eff = params.g*delta_a*params.eps/(delta_a**2 + params.kappa**2)
    eps_eff = -(eps_1_eff + 1j*eps_2_eff)
    params.g = 0.0
    params.eps = eps_eff
    params.kappa = 0.0
    params.gamma = kappa_eff
    params.c_levels = 1
    params.fd = params.f01 - delta_eff
    return params


def liouvillian_sim_alt(job_index, output_directory='./results', eigenvalue=None, eigenstate=None, eliminated=False, transmon=True):

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

    print(stack_directory)

    directory = stack_directory + '/' + sys_params.group_folder + '/' + str(sys_params.job_index)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cwd = os.getcwd()
    os.chdir(directory)
    print(directory)
    sys_params.to_csv('settings.csv')

    if not eliminated:
        if transmon:
            packaged_params = Parameters(frame_params.fc, frame_params.Ej, frame_params.g, frame_params.Ec, frame_params.eps,
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
    else:
        packaged_params = Parameters(frame_params.fc, None, frame_params.g, None, frame_params.eps,
                                     frame_params.fd, frame_params.kappa, frame_params.gamma, frame_params.t_levels,
                                     frame_params.c_levels, frame_params.gamma_phi, kappa_phi, frame_params.n_t,
                                     frame_params.n_c, frame_params.f01, frame_params.chi)
        eliminated_params = eliminate(packaged_params)
        H = hamiltonian_eliminated(eliminated_params)
        c_ops = collapse_operators(eliminated_params)

    L = liouvillian(H, c_ops)

    data = L.data
    csc = data.tocsc()

    if eigenstate is not None:
        if csc.shape[0] != eigenstate.shape[0]:
            eigenstate = None
            eigenvalue = default_eigenvalue

    k = 10
    eigvalues, states = lin.eigs(csc, k=k, sigma=eigenvalue, v0=eigenstate)
    sort_indices = np.argsort(np.abs(eigvalues))
    eigvalues = eigvalues[sort_indices]
    states = states[:,sort_indices]

    values = pd.DataFrame(eigvalues)
    values.columns = ['eigenvalues']
    states = pd.DataFrame(states)
    values.to_csv('eigenvalues.csv', index=False)


    attempts = 0
    written = False
    while not written and attempts < 3:
        try:
            states.iloc[:,0:3].to_hdf('states.h5', 'states', mode='w')
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
    os.chdir(stack_directory)


    n = packaged_params.t_levels * packaged_params.c_levels
    dims = [packaged_params.c_levels, packaged_params.t_levels]
    ground_state_vector = states.values[:, 0]
    data = dense2D_to_fastcsr_fmode(np.asfortranarray(vec2mat(ground_state_vector)), n, n)
    rho = Qobj(data, dims=[dims, dims], isherm=True)
    rho = rho + rho.dag()
    rho /= rho.tr()


    a = tensor(destroy(packaged_params.c_levels), qeye(packaged_params.t_levels))
    b = tensor(qeye(packaged_params.c_levels), destroy(packaged_params.t_levels))
    dims = a.dims
    a_exp_point = expect(a, rho)
    b_exp_point = expect(b, rho)

    num_op_a = a.dag()*a
    num_op_a.dims = dims
    num_op_b = b.dag()*b
    num_op_b.dims = dims
    n_a_exp_point = expect(num_op_a, rho)
    n_b_exp_point = expect(num_op_b, rho)

    a_exp_point = pd.DataFrame([a_exp_point], index=mi[0:1])
    b_exp_point = pd.DataFrame([b_exp_point], index=mi[0:1])
    n_a_exp_point = pd.DataFrame([n_a_exp_point], index=mi[0:1])
    n_b_exp_point = pd.DataFrame([n_b_exp_point], index=mi[0:1])
    values_frame = pd.DataFrame([eigvalues], index=mi[0:1])

    hdf_append('results.h5', a_exp_point, 'a')
    hdf_append('results.h5', b_exp_point, 'b')
    hdf_append('results.h5', n_a_exp_point, 'n_a')
    hdf_append('results.h5', n_b_exp_point, 'n_b')
    hdf_append('results.h5', values_frame, 'eigenvalues')

    os.chdir(cwd)

    return values.values[chosen_index,0], states.values[:,chosen_index]


