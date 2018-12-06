from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('-n', '--n_threads', default=1, type=int)
    args = parser.parse_args()
    output_directory = args.output_directory
    n_threads = args.n_threads 
    kwargs = dict()
    kwargs['output_directory'] = output_directory
    kwargs['save_state'] = True
    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)
        stack_frame = stack_frame.set_index('job_index')

    n_jobs = stack_frame.shape[0]

    output = parallel_map(spectrum_calc, range(n_jobs), task_kwargs=kwargs, num_cpus=n_threads)
