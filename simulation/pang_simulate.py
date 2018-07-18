import os
import argparse
import shutil
import subprocess
import pandas as pd
import glob
from cqed_tools.simulation import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('-o', '--output', default='./results', type=str)
    parser.add_argument('-r', '--resume', default=False, type=str2bool)
    parser.add_argument('-s', '--stack', default='stack.csv', type=str)
    parser.add_argument('-f', '--function', default='spectrum', type=str)
    parser.add_argument('-b', '--bistable_initial', default=str2bool)
    parser.add_argument('-n', '--n_threads', default=72, type=int)
    args = parser.parse_args()
    output_directory = args.output
    output_directory = os.path.abspath(output_directory)
    stack_path = args.stack
    resume = args.resume
    function = args.function
    bistable_initial = args.bistable_initial
    n_threads = args.n_threads
    print('In pang_simulate.py we have bistable_intitial = ' + str(bistable_initial))

    with open(stack_path, 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack = pd.read_csv(f)

    stack_directory = output_directory + '/' + stack_name

    if os.path.exists(stack_directory):
        if resume:
            try:
                os.remove(stack_directory+'/register.csv')
                print('Deleted register and resuming.')
            except:
                print('Resuming.')
        else:
            raise RuntimeError(stack_directory + ' already exists but resume = False.')
    else:
        os.mkdir(stack_directory)
        shutil.copyfile(stack_path, stack_directory+'/stack.csv')
        source_dir = '.'
        dest_dir = stack_directory
        files = glob.iglob(os.path.join(source_dir, "*.py"))
        for file in files:
            if os.path.isfile(file):
                shutil.copy2(file, dest_dir)

    os.chdir(stack_directory)

    command = 'mpiexec -n ' + str(n_threads) + ' python ' + function + '.py ' + stack_directory
    if function == 'slowdown':
        command += ' -b ' + str(bistable_initial)
    cwd = os.getcwd()

    print('cwd: ' + cwd)
    print('running in bash: ' + command)

    os.system(command)
