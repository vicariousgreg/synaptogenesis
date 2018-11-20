#ifdef __MPI__

#include <mpi.h>
#include <vector>

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

static std::vector<MPI_Request*> requests;

int mpi_wrap_create_request() {
    int id = requests.size();

    MPI_Request* req = new MPI_Request;
    *req = MPI_REQUEST_NULL;
    requests.push_back(req);

    return id;
}

void mpi_wrap_delete_request(int id) {
    if (id < 0 or id >= requests.size())
        LOG_ERROR("Attempted to delete non-existent MPI request!");

    delete requests[id];
    requests[id] = NULL;
}

void mpi_wrap_isend(int id, const void *buf, int count, int dest, int tag) {
    if (id < 0 or id >= requests.size())
        LOG_ERROR("Attempted to send using non-existent MPI request!");
    MPI_Request* req = requests[id];

    if (MPI_SUCCESS !=
            MPI_Isend(buf, count, MPI_FLOAT, dest, tag, MPI_COMM_WORLD, req))
        LOG_ERROR("Failed to call MPI_Isend!");
}

void mpi_wrap_wait(int id) {
    if (id < 0 or id >= requests.size())
        LOG_ERROR("Attempted to wait for non-existent MPI request!");
    MPI_Request* req = requests[id];
    if (MPI_SUCCESS != MPI_Wait(req, MPI_STATUS_IGNORE))
        LOG_ERROR("Failed to call MPI_Wait!");
}

void mpi_wrap_recv(void *buf, int count, int source, int tag) {
    if (MPI_SUCCESS !=
            MPI_Recv(buf, count, MPI_FLOAT, source, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE))
        LOG_ERROR("Failed to call MPI_Recv!");
}

#endif
