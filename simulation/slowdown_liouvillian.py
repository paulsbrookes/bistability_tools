from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()
    output_directory = args.output_directory
    kwargs = dict()
    kwargs['output_directory'] = output_directory

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    n_jobs = stack_frame.shape[0]

    for job_index in range(n_jobs):
        liouvillian_sim(job_index, output_directory)
