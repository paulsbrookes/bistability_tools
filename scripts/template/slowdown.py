from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':
    print('Running slowdown.py')
    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('-b', '--bistable_initial', default=True, type=str2bool)
    parser.add_argument('-t', '--transformation', default=False, type=str2bool)
    args = parser.parse_args()
    output_directory = args.output_directory
    bistable_initial = args.bistable_initial
    transformation = args.transformation
    args = []
    kwargs = dict()
    kwargs['output_directory'] = output_directory
    kwargs['bistable_initial'] = bistable_initial
    kwargs['transformation'] = transformation
    mpi_allocator(slowdown_sim, args, kwargs)
