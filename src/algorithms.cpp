#include "algorithms.hpp"

#include "global.hpp"
#include "mpi_util.hpp"

using namespace Global;

void naive_multiplication() {
    if (MPI::processID == 0) {
        C = A * B;
    }
}

void row_multiplication() {
    // Share matrix dimensions
    MPI_Bcast(&A.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    B.rows = A.cols;
    MPI_Bcast(&B.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    C.rows = A.rows;
    C.cols = B.cols;

    // Calculate base count of rows per each process
    int localRows = A.rows / MPI::processCount;
    int extraRows = A.rows % MPI::processCount;

    // Distribute extra rows between processes and calculate counts and displacements for scattering
    int *counts = new int[MPI::processCount];
    int *displacements = new int[MPI::processCount];
    for (int pID = 0; pID < MPI::processCount; ++pID) {
        counts[pID] = (localRows + (pID < extraRows)) * A.cols;
        if (pID > 0)
            displacements[pID] = displacements[pID - 1] + counts[pID - 1];
    }
    localRows += MPI::processID < extraRows;

    // Create a local buffer and distribute data between processes
    Matrix localA(localRows, A.cols);
    MPI_Scatterv(A.data, counts, displacements, MPI_DOUBLE,
                 localA.data, localA.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    Matrix localB(B.rows, B.cols);
    if (MPI::processID == 0)
        localB = B;
    MPI_Bcast(localB.data, localB.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate local result naively
    Matrix result = localA * localB;

    // Calculate result counts and displacements and gather local results
    int *resultCounts = new int[MPI::processCount];
    int *resultDisplacements = new int[MPI::processCount];
    for (int pID = 0; pID < MPI::processCount; ++pID) {
        resultCounts[pID] = counts[pID] / A.cols * B.cols;
        if (pID > 0)
            resultDisplacements[pID] = resultDisplacements[pID - 1] + resultCounts[pID - 1];
    }
    MPI_Gatherv(result.data, result.size(), MPI_DOUBLE,
                C.data, resultCounts, resultDisplacements, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    delete[] counts, delete[] displacements;
    delete[] resultCounts, delete[] resultDisplacements;
}

void column_multiplication() {
    // Share matrix dimensions
    MPI_Bcast(&A.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    B.rows = A.cols;
    MPI_Bcast(&B.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    C.rows = A.rows;
    C.cols = B.cols;

    // Transpose matrix A and scatter columns as rows of transposed matrix
    A = A.transposed();

    // Calculate base count of rows per each process
    int localRows = A.rows / MPI::processCount;
    int extraRows = A.rows % MPI::processCount;

    // Scatter rows from matrix A
    int *counts = new int[MPI::processCount];
    int *displacements = new int[MPI::processCount];
    for (int pID = 0; pID < MPI::processCount; ++pID) {
        counts[pID] = (localRows + (pID < extraRows)) * A.cols;
        if (pID > 0)
            displacements[pID] = displacements[pID - 1] + counts[pID - 1];
    }
    localRows += MPI::processID < extraRows;
    Matrix localA(A.rows, A.cols);
    MPI_Scatterv(A.data, counts, displacements, MPI_DOUBLE,
                 localA.data + displacements[MPI::processID], localRows * A.cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Scatter rows from matrix B
    int *Bcounts = new int[MPI::processCount];
    int *Bdisplacements = new int[MPI::processCount];
    for (int i = 0; i < MPI::processCount; i++)
        for (int pID = 0; pID < MPI::processCount; pID++) {
            Bcounts[pID] = counts[pID] / A.cols * B.cols;
            if (pID > 0)
                Bdisplacements[pID] = Bdisplacements[pID - 1] + Bcounts[pID - 1];
        }
    Matrix localB(B.rows, B.cols);
    MPI_Scatterv(B.data, Bcounts, Bdisplacements, MPI_DOUBLE,
                 localB.data + Bdisplacements[MPI::processID], Bcounts[MPI::processID], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Calculate local result naively
    int lCol = Bdisplacements[MPI::processID] / B.cols;
    int rCol = lCol + localRows;
    Matrix localC(C.rows, C.cols);
    for (int k = 0; k < C.cols; k++)
        for (int i = 0; i < C.rows; i++)
            for (int j = lCol; j < rCol; j++)
                localC(i, k) += localA.T(i, j) * localB(j, k); // A is transposed, so use transposed indexing

    // Reduce local results
    MPI_Reduce(localC.data, C.data, C.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] counts, delete[] displacements;
    delete[] Bcounts, delete[] Bdisplacements;
}

void block_multiplication() {
    // Abort if process count is not a square
    int blocks = MPI::processCount;
    int gridSize = int(sqrt(blocks));
    if (MPI::processID == 0 && gridSize * gridSize != blocks) {
        \
        cerr << "The process count must be a square for block multiplication!\n";
        MPI::abort();
    }

    // Share matrix dimensions
    MPI_Bcast(&A.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    B.rows = A.cols;
    MPI_Bcast(&B.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    C.rows = A.rows;
    C.cols = B.cols;

    // Calculate how many rows and columns each block is
    int blockRows = A.rows / gridSize, extraRows = A.rows % gridSize;
    int blockCols = A.cols / gridSize, extraCols = A.cols % gridSize;

    // Determine the upper left corner of each block
    int *gridRows = new int[blocks];
    int *gridCols = new int[blocks];
    for (int block = 0; block < blocks; ++block) {
        int gridRow = block / gridSize;
        int gridCol = block % gridSize;

        gridRows[block] = gridRow * blockRows + min(extraRows, gridRow);
        gridCols[block] = gridCol * blockCols + min(extraCols, gridCol);
    }

    // Reorder data by the blocks – only on the main process
    double *blockedData = new double[A.size()];
    int *blockCounts = new int[blocks];
    int *blockDisplacements = new int[blocks];
    if (MPI::processID == 0) {
        int cur = 0;
        for (int block = 0; block < blocks; block++) {
            int gridRow = block / gridSize;
            int gridCol = block % gridSize;

            // Upper left corner
            int r1 = gridRows[block];
            int r2 = r1 + blockRows + (gridRow < extraRows);

            // Lower right corner
            int c1 = gridCols[block];
            int c2 = c1 + blockCols + (gridCol < extraCols);

            blockDisplacements[block] = cur;

            for (int i = r1; i < r2; i++)
                for (int j = c1; j < c2; j++)
                    blockedData[cur++] = A(i, j);

            blockCounts[block] = cur - blockDisplacements[block];
        }
    }
    MPI::barrier();

    // Rows and columns for this process
    int localRows = blockRows + (MPI::processID / gridSize < extraRows);
    int localCols = blockCols + (MPI::processID % gridSize < extraCols);

    // Scatter the blocks
    Matrix localBlock(localRows, localCols);
    MPI_Scatterv(blockedData, blockCounts, blockDisplacements, MPI_DOUBLE,
                 localBlock.data, localBlock.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Calculate and scatter corresponding rows of matrix B
    int *Bcounts = new int[blocks];
    int *Bdisplacements = new int[blocks];
    for (int block = 0; block < blocks; block++) {
        Bcounts[block] = (blockCols + (block % gridSize < extraCols)) * B.cols;
        Bdisplacements[block] = gridCols[block] * B.cols;
    }
    Matrix localB(B.rows, B.cols);
    MPI_Scatterv(B.data, Bcounts, Bdisplacements, MPI_DOUBLE,
                 localB.data + Bdisplacements[MPI::processID], Bcounts[MPI::processID], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Calculate local result naively
    Matrix localC(C.rows, C.cols);
    for (int k = 0; k < C.cols; k++)
        for (int i = 0; i < localBlock.rows; i++)
            for (int j = 0; j < localBlock.cols; j++)
                localC(i + gridRows[MPI::processID], k) += localBlock(i, j) * localB(j + gridCols[MPI::processID], k);

    // Reduce local results
    MPI_Reduce(localC.data, C.data, C.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] gridRows, delete[] gridCols;
    delete[] blockedData;
    delete[] blockCounts, delete[] blockDisplacements;
}

void cannon_algorithm() {
    // Abort if process count is not a square
    int blocks = MPI::processCount;
    int gridSize = int(sqrt(blocks));
    if (MPI::processID == 0 && gridSize * gridSize != blocks) {
        cerr << "The process count must be a square for block multiplication!\n";
        MPI::abort();
    }

    // Share matrix dimensions
    MPI_Bcast(&A.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    B.rows = A.cols;
    MPI_Bcast(&B.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    C.rows = A.rows;
    C.cols = B.cols;

    // Abort if A or B is not a square matrix, if their sizes are different or the size is not divisible by gridSize
    if (MPI::processID == 0 && (A.rows != A.cols || B.rows != B.cols || C.rows != C.cols || (A.rows % gridSize) != 0)) {
        cerr << "Matrices must be square and of the same size for Cannon's algorithm!\n";
        MPI::abort();
    }
    int N = A.rows;

    // Calculate size of each block
    int blockSize = N / gridSize;

    // Determine the upper left corner of each block
    int *gridRows = new int[blocks];
    int *gridCols = new int[blocks];
    for (int block = 0; block < blocks; ++block) {
        int gridRow = block / gridSize;
        int gridCol = block % gridSize;

        gridRows[block] = gridRow * blockSize;
        gridCols[block] = gridCol * blockSize;
    }

    // Reorder data by the blocks – only on the main process
    double *blockedDataA = new double[N * N];
    double *blockedDataB = new double[N * N];
    int *blockCounts = new int[blocks];
    int *blockDisplacements = new int[blocks];
    if (MPI::processID == 0) {
        int cur = 0;
        for (int block = 0; block < blocks; block++) {
            // Upper left corner
            int r1 = gridRows[block];
            int r2 = r1 + blockSize;

            // Lower right corner
            int c1 = gridCols[block];
            int c2 = c1 + blockSize;

            blockDisplacements[block] = cur;

            for (int i = r1; i < r2; i++)
                for (int j = c1; j < c2; j++) {
                    blockedDataA[cur] = A(i, j);
                    blockedDataB[cur++] = B(i, j);
                }

            blockCounts[block] = cur - blockDisplacements[block];
        }
    }
    MPI::barrier();

    // Scatter the blocks
    Matrix localBlockA(blockSize), localBlockB(blockSize);
    MPI_Scatterv(blockedDataA, blockCounts, blockDisplacements, MPI_DOUBLE,
                 localBlockA.data, localBlockA.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(blockedDataB, blockCounts, blockDisplacements, MPI_DOUBLE,
                 localBlockB.data, localBlockB.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Get position of block in the grid
    int gridRow = MPI::processID / gridSize;
    int gridCol = MPI::processID % gridSize;

    // Get the block numbers for initial shift
    int nextBlockA = gridRow * gridSize + (gridCol + gridSize - gridRow) % gridSize;
    int prevBlockA = gridRow * gridSize + (gridCol + gridRow) % gridSize;

    int nextBlockB = (gridRow + gridSize - gridCol) % gridSize * gridSize + gridCol;
    int prevBlockB = (gridRow + gridCol) % gridSize * gridSize + gridCol;

    // Perform initial cycle shift
    MPI_Sendrecv_replace(localBlockA.data, blockSize * blockSize, MPI_DOUBLE,
                         nextBlockA, 0,
                         prevBlockA, MPI_ANY_TAG,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(localBlockB.data, blockSize * blockSize, MPI_DOUBLE,
                         nextBlockB, 1,
                         prevBlockB, MPI_ANY_TAG,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Get the block numbers for consecutive shifts
    nextBlockA = gridRow * gridSize + (gridCol + gridSize - 1) % gridSize;
    prevBlockA = gridRow * gridSize + (gridCol + 1) % gridSize;

    nextBlockB = (gridRow + gridSize - 1) % gridSize * gridSize + gridCol;
    prevBlockB = (gridRow + 1) % gridSize * gridSize + gridCol;

    // Start calculation cycle
    Matrix localC(N);
    for (int s = 0; s < gridSize; s++) {
        // Compute local result naively
        for (int k = 0; k < blockSize; k++)
            for (int i = 0; i < blockSize; i++)
                for (int j = 0; j < blockSize; j++)
                    localC(i + gridRows[MPI::processID], k + gridCols[MPI::processID]) +=
                            localBlockA(i, j) * localBlockB(j, k);

        // Perform cycle shift
        MPI_Sendrecv_replace(localBlockA.data, blockSize * blockSize, MPI_DOUBLE,
                             nextBlockA, 0,
                             prevBlockA, MPI_ANY_TAG,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localBlockB.data, blockSize * blockSize, MPI_DOUBLE,
                             nextBlockB, 1,
                             prevBlockB, MPI_ANY_TAG,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Reduce local results
    MPI_Reduce(localC.data, C.data, C.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] gridRows, delete[] gridCols;
    delete[] blockedDataA, delete[] blockedDataB;
    delete[] blockCounts, delete[] blockDisplacements;
}