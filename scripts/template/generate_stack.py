import pandas as pd
import numpy as np
from qutip import qload
from cqed_tools.simulation import fd1_func, fd2_func, frequencies_gen, t_gen, c_gen
from cqed_tools.mf import mf_characterise
from copy import deepcopy
from cqed_tools.calibration import eps_calc
import os


dirpath = os.getcwd()
name = os.path.basename(dirpath)
power_list = np.array([-18, -17, -16, -15, -14, -13, -12])
group_folders = ['-18dBm', '-17dBm', '-16dBm', '-15dBm', '-14dBm', '-13dBm', '-12dBm']
n_points = 21


if __name__ == '__main__':


    power_calibration = pd.read_csv('./power_calibration.csv', header=None, index_col=0)
    eps0 = power_calibration.T['eps0'].values[0]
    params = qload('params')
    
    base_Ec = params.Ec
    base_fc = params.fc
    base_Ej = params.Ej
    base_g = params.g
    base_gamma_phi = params.gamma_phi
    base_kappa_phi = params.kappa_phi
    base_gamma = params.gamma
    base_eps = 0.0
    base_kappa = params.kappa
    base_n_t = params.n_t
    base_n_c = params.n_c
    fd = 0.0
    
    sweep_list = eps_calc(power_list, eps0)
    eps_list = sweep_list
    
    endtime_list = [3e4 for param in sweep_list]
    snapshots_list = [2001 for param in sweep_list]
    
    fd1_list = fd1_func(power_list)
    fd2_list = fd2_func(power_list)
    
    fd_array = np.linspace(10.45, 10.49, 2001)
    
    fd0_list = []
    fd3_list = []
    
    for eps in eps_list:
        params_instance = deepcopy(params)
        params_instance.eps = eps
        mf_amplitude_frame = mf_characterise(params_instance, fd_array)
        fd0_list.append(mf_amplitude_frame.dropna().index[0])
        fd3_list.append(mf_amplitude_frame.dropna().index[-1])
        
    df0_list = [0.0002 for param in sweep_list]
    df1_list = [100.0 for idx, param in enumerate(sweep_list)]
    df2_list = [100.0 for param in sweep_list]
    
    
    
    gamma_phi_list = [base_gamma_phi for param in sweep_list]
    gamma_list = [base_gamma for param in sweep_list]
    nc_list = [base_n_c for param in sweep_list]
    nt_list = [base_n_t for param in sweep_list]
    fc_list = [base_fc for param in sweep_list]
    kappa_list = [base_kappa for param in sweep_list]
    g_list = [base_g for param in sweep_list]
    Ec_list = [base_Ec for param in sweep_list]
    Ej_list = [base_Ej for param in sweep_list]
    
    eps_list = np.array(eps_list)
    t_list = t_gen(eps_list) + 2
    c_list = c_gen(eps_list) + 10
    
    content = [eps_list, fd0_list, fd1_list, fd2_list, fd3_list, df0_list, df1_list, df2_list, t_list, c_list, endtime_list, snapshots_list, group_folders, gamma_list, nc_list,kappa_list,nt_list,g_list,gamma_phi_list,Ec_list,Ej_list,fc_list]
    
    columns = ['eps', 'fd0', 'fd1', 'fd2', 'fd3', 'df0', 'df1', 'df2', 't_levels', 'c_levels', 'endtime', 'snapshots', 'group_folder','gamma', 'n_c','kappa','n_t','g','gamma_phi','Ec','Ej','fc']
    
    recipe = pd.DataFrame(content).T
    recipe.columns = columns
    
    qubit_states = np.array([1])
    
    #columns = ['eps','fd','qubit_state','t_levels','c_levels','fc','Ej','g','Ec','kappa', 'gamma', 'gamma_phi', 'n_t', 'n_c', 'end_time', 'snapshots', 'group_folder', 'completed', 'running']
    columns = ['eps','fd','qubit_state','t_levels','c_levels','fc','Ej','g','Ec','kappa', 'gamma', 'gamma_phi', 'n_t', 'n_c', 'end_time', 'snapshots', 'group_folder']
    
    
    queue_list = []
    
    for index in range(recipe.shape[0]):
        row = recipe.iloc[index,:]
        frequencies = np.arange(row.fd0, row.fd1, 0.0005)
        #frequencies = frequencies_gen(row.fd0, row.fd1, row.fd2, row.fd3, row.df0, row.df1, row.df2)
    
        #arrays = np.meshgrid(row.eps, frequencies, qubit_states, row.t_levels, row.c_levels, fc, Ej, g, Ec, kappa, gamma, gamma_phi, n_t, n_c, row.endtime, row.snapshots, 1, completed, running, indexing='ij')
        arrays = np.meshgrid(row.eps, frequencies, qubit_states, row.t_levels, row.c_levels, row.fc, row.Ej, row.g, row.Ec, row.kappa, row.gamma, row.gamma_phi, row.n_t, row.n_c, row.endtime, row.snapshots, row.group_folder, indexing='ij')
        #shape = arrays[16].shape
        #arrays[16] = np.tile(row.group_folder,shape)
        
        flattened = []
        for array in arrays:
            flattened.append(array.flatten())
        
        df = pd.DataFrame(flattened).T
        df.columns = columns
        
        queue_list.append(df)
        
    combined_queue = pd.concat(queue_list)
    combined_queue.index = np.arange(combined_queue.shape[0])
    combined_queue.index.name = 'job_index'
    
    
    
    with open('stack.csv','w') as f:
        f.write(name+'\n')
    combined_queue.to_csv('stack.csv',mode='a')
