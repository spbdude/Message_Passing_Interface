import numpy as np
from mpi4py import MPI
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    timings = {}

    Msizes = [10, 100, 1000]
    for n in Msizes:
        if rank == 0:
            start = time.time()
            A = np.random.rand(n, n)
            b = np.random.rand(n)
        else:
            A = np.zeros((n, n))
            b = np.zeros(n)

        comm.Bcast(A, root=0)
        comm.Bcast(b, root=0)

        n = len(b)
        for m in range(n):
            if m % size == rank:
                norm_r = A[m] / A[m, m]
                norm_v = b[m] / A[m, m]
                comm.Bcast(norm_r, root=rank)
                comm.bcast(norm_v, root=rank)

                for k in range(m + 1, n):
                    if (k % size) == rank:
                        A[k] -= norm_r * A[k, m]
                        b[k] -= norm_v * A[k, m]
            else:
                norm_r = np.empty(n, dtype=np.float64)
                norm_v = None
                comm.Bcast(norm_r, root=m % size)
                norm_v = comm.bcast(norm_v, root=m % size)
                for k in range(m + 1, n):
                    if k % size == rank:
                        A[k] -= norm_r * A[k, m]
                        b[k] -= norm_v * A[k, m]

        x = np.zeros(n, dtype=np.float64)
        for m in range(n - 1, -1, -1):
            if m % size == rank:
                x[m] = b[m] / A[m, m]
            x[m] = comm.bcast(x[m], root=m % size)

            for k in range(m):
                if k % size == rank:
                    b[k] -= A[k, m] * x[m]

        if rank == 0:
            endt = time.time()
            timings[n] = endt - start
            print(f"size: {n} time: {timings[n]} sec")
            print("\n")

if __name__ == "__main__":
    main()