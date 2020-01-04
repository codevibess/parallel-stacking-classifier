
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-t', '--type', help='Type of program run, available 2 types: sequencial and parallel', required=True, choices=['sequence','parallel'], default='parallel')
required.add_argument('-d', '--dataset', help='Dataset', required=True, choices=['MNIST','CIFAR-10', 'CIFAR-100', 'letter-recognition'])
required.add_argument('-m', '--method', help='Method: CV train-test', required=True, choices=['CV','test-train'])
parser.parse_args()

args = parser.parse_args(sys.argv[1:])
print(args.type)

if args.type == 'sequence':
    process = subprocess.Popen("python sequencial_flow.py", shell=True)
elif args.type == 'parallel':
    process = subprocess.Popen(f"mpiexec python -m mpi4py parallel_flow.py {args.dataset} {args.method}" , shell=True)



