#include "global.hpp"

namespace Global {

    Matrix A, B, C;
    ifstream fin;
    ofstream fout;

    void init(const string &input_file) {
        fin.open(input_file);
        fout.open("output.txt");
        int rows, cols;

        fin >> rows >> cols;
        A = Matrix(rows, cols);
        fin >> A;

        fin >> rows >> cols;
        B = Matrix(rows, cols);
        fin >> B;

        if (A.cols != B.rows) {
            cerr << "Sizes do not match!!!";
            exit(1);
        }

        C = Matrix(A.rows, B.cols);
    }

    void finalize() {
        fout << C;

        fin.close();
        fout.close();
    }

}  // namespace Global