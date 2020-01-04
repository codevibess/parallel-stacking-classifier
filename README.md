# Parallel Stacking Classifier

Implementation of parallel computing stacking classifier using Message Passing Interface.
Stacking classifier is based on 4 classifiers:
- GaussianNB
- RandomForestClassifier
- SVC
- KNeighborsClassifier

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

3) Run:

> Note that you need to specify parameters like -t (type of run), -m (method), -d (dataset name)

Example invocation:
```
python main.py -t parallel -m test-train -d MNIST -n 4 
```

Available types:
- sequence (runned by python)
- parallel (runned by mpiexec)

Available methods:
- CV
- test-train

Available datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- letter-recognition

> program will automatically create process using formula: 1 physical core = 1 process

>> program runned with module mpi4py to avoid deadlocks ( the finalizer hook of mpi4py module will call MPI_Abort() on the MPI_COMM_WORLD communicator, thus effectively aborting the MPI execution environment.)