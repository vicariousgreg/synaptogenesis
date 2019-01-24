#ifdef __MPI__

#include <mpi.h>
#include <set>
#include <vector>
#include <cstdlib>

#include "mpi_wrap.h"
#include "util/logger.h"

void mpi_wrap_init() {
    int prov;
    if (MPI_SUCCESS !=
			MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &prov))
        LOG_ERROR("MPI_Init_thread returned an error!");
}

void mpi_wrap_finalize() {
    if (MPI_SUCCESS != MPI_Finalize())
        LOG_ERROR("MPI_Finalize returned an error!");
}

int mpi_wrap_get_rank() {
    int mpi_rank;
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank))
        LOG_ERROR("MPI_Comm_rank returned an error!");
    return mpi_rank;
}

int mpi_wrap_get_size() {
    int mpi_size;
    if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &mpi_size))
        LOG_ERROR("MPI_Comm_size returned an error!");
    return mpi_size;
}

void mpi_wrap_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

static std::set<MPI_Request*> requests;
static std::vector<MPI_Request*> request_arrays;

void* mpi_wrap_create_request() {
    MPI_Request* req = new MPI_Request;
    *req = MPI_REQUEST_NULL;
    requests.insert(req);

    return req;
}

int mpi_wrap_create_request_array(int size) {
    if (size < 0)
        LOG_ERROR("Attempted to create negative size MPI request array!");
    MPI_Request* reqs = (MPI_Request*)malloc(size * sizeof(MPI_Request));
    for (int i = 0 ; i < size ; ++i)
        *(reqs+i) = MPI_REQUEST_NULL;

    int index = request_arrays.size();
    request_arrays.push_back(reqs);

    return index;
}

void mpi_wrap_delete_request(void* vreq) {
    MPI_Request* req = (MPI_Request*)vreq;
    if (requests.find(req) == requests.end())
        LOG_ERROR("Attempted to delete non-existent MPI request!");

    delete req;
    requests.erase(req);
}

void mpi_wrap_delete_request_array(int index) {
    if (index < request_arrays.size())
        LOG_ERROR("Attempted to delete non-existent MPI request array!");

    free(request_arrays[index]);
    request_arrays[index] = NULL;
}

void mpi_wrap_isend(void* vreq, const void *buf, int count, int dest, int tag) {
    MPI_Request* req = (MPI_Request*)vreq;
    if (requests.find(req) == requests.end())
        LOG_ERROR("Attempted to send using non-existent MPI request!");

    if (MPI_SUCCESS !=
            MPI_Isend(buf, count, MPI_FLOAT, dest, tag, MPI_COMM_WORLD, req))
        LOG_ERROR("Failed to call MPI_Isend!");
}

void mpi_wrap_wait(void* vreq) {
    MPI_Request* req = (MPI_Request*)vreq;
    if (requests.find(req) == requests.end())
        LOG_ERROR("Attempted to wait for non-existent MPI request!");

    if (MPI_SUCCESS != MPI_Wait(req, MPI_STATUS_IGNORE))
        LOG_ERROR("Failed to call MPI_Wait!");
}

int mpi_wrap_wait_any(int size, int arr) {
    int index;

    if (MPI_SUCCESS !=
            MPI_Waitany(size, request_arrays[arr],
                &index, MPI_STATUS_IGNORE))
        LOG_ERROR("Failed to call MPI_Waitany!");

    return index;
}

void mpi_wrap_recv(void *buf, int count, int source, int tag) {
    if (MPI_SUCCESS !=
            MPI_Recv(buf, count, MPI_FLOAT, source, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE))
        LOG_ERROR("Failed to call MPI_Recv!");
}

void mpi_wrap_irecv(int arr, int index, void *buf, int count, int source, int tag) {
    if (MPI_SUCCESS !=
            MPI_Irecv(buf, count, MPI_FLOAT, source, tag,
                MPI_COMM_WORLD, request_arrays[arr] + index))
        LOG_ERROR("Failed to call MPI_Irecv!");
}

#endif
