from mpi4py import MPI
import numpy as np
import time


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    num_elements = [10, 1000, 10000000]
    timings = {}

    for N in num_elements:
        data = None
        if rank == 0:
            data = np.random.rand(N)
            start_time = time.time()


        part = np.empty(N // size, dtype='d')
        comm.Scatter(data, part, root=0)

        local_sum = np.sum(part)
        total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

        if rank == 0:
            end_time = time.time()
            timings[N] = end_time - start_time
            print(f" size: {N} total sum: {total_sum}, time: {timings[N]} sec")
            print("\n")




if __name__ == '__main__':
    main()


