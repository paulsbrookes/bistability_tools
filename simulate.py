from tools.spectrum_calc import *
from tools.slowdown_sim import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()
    output_directory = args.output_directory
    mpi_allocator(slowdown_sim, output_directory)
