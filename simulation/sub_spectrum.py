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
        shutil.copyfile('slowdown.py', stack_directory + '/slowdown.py')
        shutil.copyfile('sub_slowdown.py', stack_directory+'/sub_slowdown.py')
        shutil.copyfile('sub_spectrum.py', stack_directory+'/sub_spectrum.py')
        #shutil.copytree('tools', stack_directory+'/tools')

    n_threads = 48

    content = "#!/bin/bash -l\n\n" \
    "# Batch script to run an OpenMP threaded job on Legion with the upgraded\n" \
    "# software stack under SGE.\n\n" \
    "# 1. Force bash as the executing shell.\n" \
    "#$ -S /bin/bash\n\n" \
    "# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).\n" \
    "#$ -l h_rt=12:0:0\n\n" \
    "# 3. Request 1 gigabyte of RAM for each core/thread\n" \
    "#$ -l mem=0.5G\n\n" \
    "# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)\n" \
    "#$ -l tmpfs=0.5G\n\n" \
    "# 6. Select 1 thread.\n" \
    "#$ -pe mpi " + str(n_threads) + " \n\n" \
    "# 7. Set the working directory to somewhere in your scratch space. This is\n" \
    "# a necessary step with the upgraded software stack as compute nodes cannot\n" \
    "# write to \$HOME.\n" \
    "# Replace \"<your_UCL_id>\" with your UCL user ID :)\n" \
    "#$ -wd " + stack_directory + "\n\n" \
    "module unload compilers/intel/2017/update1\n" \
    "module unload mpi/intel/2017/update1/intel\n" \
    "module load python2/recommended\n" \
    "module load compilers/gnu/4.9.2\n" \
    "module load mpi/openmpi/1.10.1/gnu-4.9.2\n" \
    "module load mpi4py/2.0.0/python2\n" \
    "module load qutip/4.1.0/python-2.7.12\n\n" \
    "# 8. Run the application.\n" \
    "mpiexec -n " + str(n_threads) + "  python spectrum.py " + stack_directory

    text_file = open("./submit.sh", "w")
    text_file.write(content)
    text_file.close()

    n_rounds = 5
    round_names = [stack_name+"_"+str(i) for i in range(n_rounds)]
    subprocess.call(["qsub","-N",round_names[0], "./submit.sh"])
    for i in range(1,n_rounds):
        subprocess.call(["qsub","-N",round_names[i],"-hold_jid",round_names[i-1], "./submit.sh"])