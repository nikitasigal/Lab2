#include <unistd.h>

#include "algorithms.hpp"
#include "global.hpp"
#include "matrix.hpp"
#include "mpi_util.hpp"

#include <iomanip>

using namespace std;

int main(int argc, char **argv) {
    MPI::init(&argc, &argv);

    if (MPI::processID == 0) {
        if (argc != 3) {
            cerr << "Usage: " << argv[0] << " <input_file> <algorithm>\n";
            MPI::abort();
        }

        Global::init(argv[1]);
    }

    MPI::barrier();

    double startTime;
    if (MPI::processID == 0) {
        startTime = MPI_Wtime();
    }

    string algorithm = string(argv[2]);
    if (algorithm == "naive") {
        naive_multiplication();
    } else if (algorithm == "row") {
        row_multiplication();
    } else if (algorithm == "column") {
        column_multiplication();
    } else if (algorithm == "block") {
        block_multiplication();
    } else if (algorithm == "cannon") {
        cannon_algorithm();
    } else {
        cerr << "Unknown algorithm\n";
    }

    MPI::barrier();
    if (MPI::processID == 0) {
        double endTime = MPI_Wtime();
        cout << fixed << setprecision(6) << endTime - startTime << '\n';

        Global::finalize();
    }

    MPI::finalize();

    return 0;
}