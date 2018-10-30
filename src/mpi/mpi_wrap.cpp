#include <mpi.h>

#include "mpi_wrap.h"

void mpi_wrap_init() {
    printf("Initializing MPI...\n");
    MPI::Init_thread(MPI_THREAD_MULTIPLE);
}

int mpi_get_rank() {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    return mpi_rank;
}

void mpi_wrap_finalize() {
    MPI_Finalize();
}
