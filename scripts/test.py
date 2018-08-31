import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify output directory.')
    parser.add_argument('-b', '--bistable_initial', default=True, type=str2bool)
    args = parser.parse_args()
    print(args.bistable_initial)

