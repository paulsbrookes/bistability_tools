import os
import argparse
import shutil
import subprocess
import pandas as pd
import glob


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
	    try:
            	os.remove(stack_directory+'/register.csv')
            	print('Deleted register and resuming.')
	    except:
                print('Resuming.')
        else:
            raise RuntimeError(stack_directory + ' already exists but resume = False.')
    else:
        os.mkdir(stack_directory)
	source_dir = '.'
	dest_dir = stack_directory
	files = glob.iglob(os.path.join(source_dir, "*.py"))
	for file in files:
	    if os.path.isfile(file):
		shutil.copy2(file, dest_dir)


    content = "#! /bin/bash \n\n" \
    "# Batch script to run an OpenMP threaded job on Legion with the upgraded\n" \
    "# software stack under SGE.\n\n" \
    "# 1. Force bash as the executing shell.\n" \
    "#$ -S /bin/bash\n\n" \
    "#$ -t 1-307\n" \
    "# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).\n" \
    "#$ -l h_rt=0:10:0\n\n" \
    "# 3. Request 1 gigabyte of RAM for each core/thread\n" \
    "#$ -l mem=1G\n\n" \
    "# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)\n" \
    "#$ -l tmpfs=1G\n\n" \
    "# 6. Select 1 thread.\n" \
    "#$ -pe smp 1\n\n" \
    "# 7. Set the working directory to somewhere in your scratch space. This is\n" \
    "# a necessary step with the upgraded software stack as compute nodes cannot\n" \
    "# write to \$HOME.\n" \
    "# Replace \"<your_UCL_id>\" with your UCL user ID :)\n" \
    "#$ -wd " + stack_directory + "\n\n" \
    "module unload compilers\n" \
    "module load compilers/gnu\n" \
    "module unload mpi/intel/2017/update1/intel\n" \
    "module load mpi/openmpi/1.10.1/gnu-4.9.2\n" \
    "conda activate bistable\n" \
    "module load gsl/2.4/gnu-4.9.2\n\n" \
    "# 8. Run the application.\n" \
    "python slowdown_array.py " + stack_directory + " $SGE_TASK_ID"

    text_file = open("./submit.sh", "w")
    text_file.write(content)
    text_file.close()

    n_rounds = 1
    round_names = [stack_name+"_"+str(i) for i in range(n_rounds)]
    subprocess.call(["qsub","-N",round_names[0], "./submit.sh"])
    for i in range(1,n_rounds):
        subprocess.call(["qsub","-N",round_names[i],"-hold_jid",round_names[i-1], "./submit.sh"])
