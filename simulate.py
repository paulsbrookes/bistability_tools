from tools.legion_tools import *
from tools.slowdown_sim import *


if __name__ == '__main__':
    home = os.environ['HOME']
    output_directory = home + '/Scratch/register'
    mpi_allocator(slowdown_sim, output_directory)
