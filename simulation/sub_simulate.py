import os
import argparse
import shutil
import subprocess
import pandas as pd
import glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('-o', '--output', default='..', type=str)
    parser.add_argument('-r', '--resume', default=False, type=bool)
    parser.add_argument('-s', '--stack', default='stack.csv', type=str)
    parser.add_argument('-f', '--function', default='slowdown', type=str)
    parser.add_argument('-n', '--n_threads', default=32, type=int)
    parser.add_argument('-c', '--n_cycles', default=5, type=int)
    parser.add_argument('-m', '--method', default='mpi', type=str)
    parser.add_argument('-a', '--avx', default=False, type=bool)
    args = parser.parse_args()
    output_directory = args.output
    output_directory = os.path.abspath(output_directory)
    stack_path = args.stack
    resume = args.resume
    function = args.function
    n_threads = args.n_threads
    n_cycles = args.n_cycles
    method = args.method
    avx = args.avx

    with open(stack_path, 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack = pd.read_csv(f)
        n_rows = stack.shape[0]

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
        os.mkdir(stack_directory+'/output')
        shutil.copyfile(stack_path, stack_directory+'/stack.csv')
	source_dir = '.'
	dest_dir = stack_directory
	files = glob.iglob(os.path.join(source_dir, "*.py"))
	for file in files:
	    if os.path.isfile(file):
		shutil.copy2(file, dest_dir)


    content = "#!/bin/bash -l\n\n" \
    "# Batch script to run an OpenMP threaded job on Legion with the upgraded\n" \
    "# software stack under SGE.\n\n" \
    "# 1. Force bash as the executing shell.\n" \
    "#$ -S /bin/bash\n\n" \
    "# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).\n" \
    "#$ -l h_rt=12:0:0\n\n" \
    "# 3. Request 1 gigabyte of RAM for each core/thread\n" \
    "#$ -l mem=1G\n\n" \
    "# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)\n" \
    "#$ -l tmpfs=1G\n\n" \
    "#$ -o ./output\n" \
    "#$ -e ./output\n" \
    "# 7. Set the working directory to somewhere in your scratch space. This is\n" \
    "# a necessary step with the upgraded software stack as compute nodes cannot\n" \
    "# write to \$HOME.\n" \
    "# Replace \"<your_UCL_id>\" with your UCL user ID :)\n" \
    "#$ -wd " + stack_directory + "\n\n" \
    "module load gsl/2.4/intel-2017\n"

    if avx == False:
        content += "module load python2/recommended\n" \
        "module load qutip/4.1.0/python-2.7.12\n"
    else:
        content += "conda activate bistable\n" \
        "#$ -ac allow=LMNOPQSTU\n"

    if method == 'mpi':
        content += "# 6. Select required number of threads.\n" \
    	"#$ -pe mpi " + str(n_threads) + " \n\n" \
    	"# 8. Run the application.\n" \
    	"gerun python " + function + ".py " + stack_directory
    elif method == 'array':
        content += "#$ -t 1-" + str(n_rows) + "\n" \
    	"#$ -pe smp 1\n" \
        "# 8. Run the application.\n" \
        "python slowdown_array.py " + stack_directory + " $SGE_TASK_ID"

    text_file = open("./submit.sh", "w")
    text_file.write(content)
    text_file.close()

    round_names = [stack_name+"_"+str(i) for i in range(n_cycles)]
    subprocess.call(["qsub","-N",round_names[0], "./submit.sh"])
    for i in range(1,n_cycles):
        subprocess.call(["qsub","-N",round_names[i],"-hold_jid",round_names[i-1], "./submit.sh"])
