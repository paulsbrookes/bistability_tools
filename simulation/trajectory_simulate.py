import os
import argparse
import shutil
import subprocess
import pandas as pd
import glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('-o', '--output', default='..', type=str)
    parser.add_argument('-r', '--resume',default=False)
    parser.add_argument('-s', '--stack', default='stack.csv')
    parser.add_argument('-f', '--function', default='spectrum')
    args = parser.parse_args()
    output_directory = args.output
    output_directory = os.path.abspath(output_directory)
    stack_path = args.stack
    resume = args.resume
    function = args.function

    with open(stack_path, 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack = pd.read_csv(f)

    stack_directory = output_directory + '/' + stack_name

    if os.path.exists(stack_directory):
        if resume:
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

    command = 'python ' + function + '.py ' + stack_directory
    cwd = os.getcwd()

    print('cwd: ' + cwd)
    print('running in bash: ' + command)

    os.system(command)
