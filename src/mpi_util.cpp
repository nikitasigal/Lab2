#include "mpi_util.hpp"

namespace MPI {

    int processID;
    int processCount;
    int workerCount;

    void init(int *argc, char ***argv) {
        MPI_Init(argc, argv);
        MPI_Comm_size(MPI_COMM_WORLD, &processCount);
        MPI_Comm_rank(MPI_COMM_WORLD, &processID);
        workerCount = processCount - 1;
    }

    void barrier() {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void abort() {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    void finalize() {
        MPI_Finalize();
    }

}  // namespace MPI