from cqed_tools.simulation import *
import argparse


if __name__ == '__main__':
    print('Running slowdown.py')
    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    parser.add_argument('-b', '--bistable_initial', default=True, type=str2bool)
    parser.add_argument('-t', '--transformation', default=False, type=str2bool)
    parser.add_argument('-mf', '--mf_init', default=False, type=str2bool)
    parser.add_argument('-g', '--g', default=np.sqrt(2), type=float)
    args = parser.parse_args()
    output_directory = args.output_directory
    bistable_initial = args.bistable_initial
    transformation = args.transformation
    args = []
    kwargs = dict()
    kwargs['output_directory'] = output_directory
    kwargs['bistable_initial'] = bistable_initial
    kwargs['transformation'] = transformation
    kwargs['g'] = args.g
    kwargs['mf_init'] = args.mf_init
    print('In slowdown.py we have bistable_initial = ' + str(bistable_initial))
    mpi_allocator(slowdown_sim, args, kwargs)
