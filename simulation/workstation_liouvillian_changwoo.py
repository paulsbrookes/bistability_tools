import os
import argparse
import shutil
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

    print('hello')
    print(stack_directory)
    print('hello')

    if os.path.exists(stack_directory):
        if not resume:
            raise RuntimeError(stack_directory + ' already exists but resume = False.')
    else:
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        os.mkdir(stack_directory)
        shutil.copyfile('stack.csv', stack_directory+'/stack.csv')
        shutil.copyfile('slowdown_liouvillian_alt.py', stack_directory + '/slowdown_liouvillian_alt.py')
        shutil.copyfile('workstation_liouvillian_changwoo.py', stack_directory+'/workstation_liouvillian_changwoo.py')

    os.chdir(stack_directory)

    command = 'python slowdown_liouvillian_alt.py ' + stack_directory + ' -e True'
    cwd = os.getcwd()

    print('cwd: ' + cwd)
    print('running in bash: ' + command)

    os.system(command)
