from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)
# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     req = comm.isend(data, dest=1, tag=11)
#     print('sended')
#     req.wait()
#     print('wait')
# elif rank == 1:
#     req = comm.irecv(source=0, tag=11)
#     print('wait from rank 1')
#     data = req.wait()
#     print(data)