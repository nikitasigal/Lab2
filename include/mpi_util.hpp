#pragma once

#include <mpi.h>

namespace MPI {

    extern int processID;
    extern int processCount;
    extern int workerCount;

    void init(int *argc, char ***argv);

    void barrier();

    void abort();

    void finalize();

}  // namespace MPI