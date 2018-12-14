#ifndef mpi_wrap_h
#define mpi_wrap_h

#ifdef __MPI__

void mpi_wrap_init();
void mpi_wrap_finalize();

int mpi_wrap_get_rank();
int mpi_wrap_get_size();

void mpi_wrap_barrier();

void* mpi_wrap_create_request();
int mpi_wrap_create_request_array(int size);
void mpi_wrap_delete_request(void* req);
void mpi_wrap_delete_request_array(int index);

void mpi_wrap_isend(void* req, const void *buf, int count, int dest, int tag);
void mpi_wrap_wait(void* req);
int mpi_wrap_wait_any(int size, int arr);
void mpi_wrap_recv(void *buf, int count, int source, int tag);
void mpi_wrap_irecv(int arr, int index, void *buf, int count, int source, int tag);

#else

inline void mpi_wrap_init() { }
inline void mpi_wrap_finalize() { }

inline int mpi_wrap_get_rank() { return 0; }
inline int mpi_wrap_get_size() { return 1; }

inline void mpi_wrap_barrier() { }

inline void* mpi_wrap_create_request() { return NULL; }
inline int mpi_wrap_create_request_array(int size) { return 0; }
inline void mpi_wrap_delete_request(void* req) { }
inline void mpi_wrap_delete_request_array(int index) { }

inline void mpi_wrap_isend(void* vreq, const void *buf, int count, int dest, int tag) { }
inline void mpi_wrap_wait(void* req) { }
inline int mpi_wrap_wait_any(int size, int arr) { return 0; }
inline void mpi_wrap_recv(void *buf, int count, int source, int tag) { }
inline void mpi_wrap_irecv(int arr, int index, void *buf, int count, int source, int tag) { }

#endif

#endif
