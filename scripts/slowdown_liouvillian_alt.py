from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('-e', '--eliminated', default=False)
    args = parser.parse_args()
    output_directory = args.output_directory
    eliminated = args.eliminated

    with open('stack.csv', 'r') as f:
        header = f.readline()
        stack_name = header.split('\n')[0]
        stack_frame = pd.read_csv(f)

    n_jobs = stack_frame.shape[0]

    eigenvalue = None
    eigenstate = None

    for job_index in range(n_jobs):
        eigenvalue, eigenstate = liouvillian_sim_alt(job_index, output_directory=output_directory, eigenvalue=eigenvalue, eigenstate=eigenstate, eliminated=eliminated)