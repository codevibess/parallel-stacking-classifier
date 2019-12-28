from mpi4py import MPI
import numpy as np

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


# MPI environment finalization
MPI.Finalize()
