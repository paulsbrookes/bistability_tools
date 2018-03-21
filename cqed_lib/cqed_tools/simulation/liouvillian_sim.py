from .legion_tools import *
import scipy.sparse.linalg as lin

def liouvillian_sim(job_index, output_directory='./results'):

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
    #directory = stack_directory + '/' + sys_params.group_folder + '/' + str(job_index)

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

    values, states = lin.eigs(csc, k=2, sigma=0)
    values = pd.DataFrame(values)
    values.columns = ['eigenvalues']
    states = pd.DataFrame(states)
    values.to_csv('eigenvalues.csv',index=False)
    states.to_csv('states.csv',index=False)

    os.chdir(cwd)
