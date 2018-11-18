#ifndef mpi_wrap_h
#define mpi_wrap_h

#ifdef __MPI__

void mpi_wrap_init();
void mpi_wrap_finalize();

int mpi_wrap_get_rank();
int mpi_wrap_get_size();

int mpi_wrap_create_request();
void mpi_wrap_delete_request(int id);
void mpi_wrap_isend(int id, const void *buf, int count, int dest, int tag);
void mpi_wrap_wait(int id);
void mpi_wrap_recv(void *buf, int count, int source, int tag);

#else

void mpi_wrap_init() { }
void mpi_wrap_finalize() { }

int mpi_wrap_get_rank() { return 0; }
int mpi_wrap_get_size() { return 1; }

int mpi_wrap_isend(const void *buf, int count, int dest, int tag) { return 0; }
void mpi_wrap_wait(int id) { }
void mpi_wrap_recv(void *buf, int count, int source, int tag) { }

#endif

#endif
