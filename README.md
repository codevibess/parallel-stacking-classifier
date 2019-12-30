# Parallel Stacking Classifier

Implementation of parallel computing stacking classifier using Message Passing Interface.
Stacking classifier is based on 4 classifiers:
- add
- add
- add
- add

Parallel computing workflow:

Master process sends the data to the slave processes, every process classifies using bounded for this process algorithm and then sends the classification results back to the master process.


## Requirement

- mpi 
- python 3
- virtual-env

### How to run ?

1) Activate Virtualenv. [Instruction](https://virtualenv.pypa.io/en/latest/userguide/)


2) Install dependecies:

```
pip install -r requirements.txt
```

3) Run using mpiexec:

```
mpiexec python -m mpi4py parallel_main.py -t type -d datasetName
```
> if you dont specify number of processes manually, program will automatically create process using formula: 1 physical core = 1 process

>> program runned with module mpi4py to avoid deadlocks ( the finalizer hook of mpi4py module will call MPI_Abort() on the MPI_COMM_WORLD communicator, thus effectively aborting the MPI execution environment.)