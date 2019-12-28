import numpy as np
from mpi4py import MPI


# initialize MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    print(f"[INFO] Program runned in {size} processes")

print(f"[INFO] Hello from process number {rank}")


if rank == 0:
    data = np.arange(100, dtype='i')
    for i in range(100):
        assert data[i] == i
else:
    data = np.empty(100, dtype='i')
    pass



print(f'[INFO] Bcasting data from the root process ({rank})') if rank == 0 else None
bcast_start_time = MPI.Wtime()
comm.Bcast(data, root=0)
bcast_finish_time = MPI.Wtime()

bcast_time = bcast_finish_time - bcast_start_time
print(f'[TIME] Master process ({rank}) finished Bcasting data with time {bcast_time}') if rank == 0 else print(f'[TIME] Process {rank} finished receive bcasted data with time {bcast_time}')


# classification 
algorithm=None
classification_time_start = MPI.Wtime()
if rank == 0:
    classification_output = "0"
    algorithm = 'knn'
    pass
elif rank == 1:
    classification_output = "1"
    algorithm='svc'
    pass
elif rank == 2:
    classification_output = "2"
    algorithm='grigsearch'
    pass
elif rank == 3:
    classification_output = "3"
    algorithm='forest'
    pass

classification_time_end = MPI.Wtime()
classification_time = classification_time_end - classification_time_start
print(f'[TIME] Process {rank} finished classification by {algorithm} algorithm with time: {classification_time}')

# stacking
outputs_from_classifications = comm.gather(classification_output)
print(outputs_from_classifications)

# MPI environment finalization
MPI.Finalize()
