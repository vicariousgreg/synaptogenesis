#ifndef mpi_wrap_h
#define mpi_wrap_h

#ifdef __MPI__

void mpi_wrap_init();
void mpi_wrap_finalize();

int mpi_wrap_get_rank();
int mpi_wrap_get_size();

void mpi_wrap_barrier();

int mpi_wrap_create_request();
void mpi_wrap_delete_request(int id);

void mpi_wrap_isend(int id, const void *buf, int count, int dest, int tag);
void mpi_wrap_wait(int id);
void mpi_wrap_recv(void *buf, int count, int source, int tag);

#else

inline void mpi_wrap_init() { }
inline void mpi_wrap_finalize() { }

inline int mpi_wrap_get_rank() { return 0; }
inline int mpi_wrap_get_size() { return 1; }

inline void mpi_wrap_barrier() { }

inline int mpi_wrap_create_request() { return 0; }
inline void mpi_wrap_delete_request(int id) { }

inline int mpi_wrap_isend(const void *buf, int count, int dest, int tag) { return 0; }
inline void mpi_wrap_wait(int id) { }
inline void mpi_wrap_recv(void *buf, int count, int source, int tag) { }

#endif

#endif
