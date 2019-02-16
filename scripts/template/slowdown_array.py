from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('-b', '--bistable_initial', default=True, type=str2bool)
    parser.add_argument('-t', '--transformation', default=False, type=str2bool)
    parser.add_argument('-mf', '--mf_init', default=False, type=str2bool)
    parser.add_argument('-g', '--g', default=np.sqrt(2), type=float)
    kwargs = dict()
    kwargs['g'] = args.g
    kwargs['mf_init'] = args.mf_init
    kwargs['transformation'] = args.transformation
    kwargs['bistable_initial'] = args.bistable_initial 
    args = parser.parse_args()
    output_directory = args.output_directory
    job_id = args.job_id
    job_index = job_id - 1
    print('Hello from job id = '+str(job_id))
    slowdown_sim(job_index, output_directory, **kwargs)


