import os
import argparse
import shutil
import subprocess
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory',type=str)
    parser.add_argument('-r','--resume',default=False)
    args = parser.parse_args()
    output_directory = args.output_directory
    output_directory = os.path.abspath(output_directory)
    resume = args.resume

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack = pd.read_csv(f)

    stack_directory = output_directory + '/' + stack_name

    if os.path.exists(stack_directory):
        if resume:
            os.remove(stack_directory+'/register.csv')
            print('Deleted register and resuming.')
        else:
            raise RuntimeError(stack_directory + ' already exists but resume = False.')
    else:
        os.mkdir(stack_directory)
        shutil.copyfile('stack.csv', stack_directory+'/stack.csv')
        shutil.copyfile('spectrum.py', stack_directory+'/spectrum.py')
        shutil.copyfile('slowdown_liouvillian.py', stack_directory + '/slowdown_liouvillian.py')
        shutil.copyfile('pang_liouvillian.py', stack_directory+'/pang_liouvillian.py')

    os.chdir(stack_directory)

    command = 'python slowdown_liouvillian.py ' + stack_directory
    cwd = os.getcwd()

    print('cwd: ' + cwd)
    print('running in bash: ' + command)

    os.system(command)
