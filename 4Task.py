from mpi4py import MPI
import numpy as np
import time


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = {}


    Msize = [10, 100, 1000]
    for n in Msize:
        # print(Msize[i])
        A = np.random.rand(n,n)
        B = np.random.rand(n,n)
        C = np.empty((n,n), dtype=np.float64)

        start_time = time.time()
        A_r, A_c = A.shape

        xtr = A_r % size
        rp = A_r // size
        if rank < xtr:
            start_r = rank * (rp + 1)
            end_r = start_r + (A_r // size) + 1
        else:
            start_r = rank * (rp) + xtr
            end_r = start_r + (rp)

        local_A = A[start_r:end_r, :]
        local_C = np.dot(local_A, B)

        if rank == 0:
            C[start_r:end_r, :] = local_C
            for i in range(1, size):
                if i < xtr:
                    proc_r = rp + 1
                else:
                    proc_r = rp
                pstart_r = i * rp + min(i, xtr)
                C[pstart_r:pstart_r + proc_r, :] = comm.recv(source=i, tag=10)
        else:
            comm.send(local_C, dest=0, tag=10)


        if rank == 0:
            end_time = time.time()
            timings[n] = end_time - start_time
            print(f"size: {n} time: {timings[n]} sec")
            print("\n")


if __name__ == "__main__":
    main()