from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('job_id', type=int)
    args = parser.parse_args()
    output_directory = args.output_directory
    job_id = args.job_id
    job_index = job_id - 1
    print('Hello from job id = '+str(job_id))
    slowdown_sim(job_index, output_directory)
