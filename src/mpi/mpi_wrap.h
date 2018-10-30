#ifndef mpi_wrap_h
#define mpi_wrap_h

#ifdef __MPI__

void mpi_wrap_init();
int mpi_get_rank();
void mpi_wrap_finalize();

#else

void mpi_wrap_init() { }
int mpi_get_rank() { return 0; }
void mpi_wrap_finalize() { }

#endif

#endif
