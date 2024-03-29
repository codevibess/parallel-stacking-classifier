
import argparse
import subprocess
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-t', '--type', 
                        help='Type of program run, available 2 types: sequencial and parallel',
                        required=True, 
                        choices=['sequence', 'parallel'], default='parallel')
    required.add_argument('-d', '--dataset', 
                        help='Dataset', 
                        required=True,
                        choices=['MNIST', 'CIFAR-10', 'CIFAR-100', 'letter-recognition'])
    required.add_argument('-m', '--method',
                        help='Method: CV train-test',
                        required=True,
                        choices=['CV', 'test-train'])
    required.add_argument('-n', '--numberOfProcesses',
                        nargs='?',
                        help='Number of processes')
    parser.parse_args()

    args = parser.parse_args(sys.argv[1:])
    print(args.type)

    if args.type == 'sequence':
        process = subprocess.Popen(
            f"python sequencial_flow.py {args.dataset} {args.method}", 
            shell=True)
    elif args.type == 'parallel':
        process = subprocess.Popen(
            f"mpiexec -n {args.numberOfProcesses} python -m mpi4py parallel_flow.py {args.dataset} {args.method}",
            shell=True)
