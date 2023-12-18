#pragma once

#include <fstream>
#include <string>

#include "matrix.hpp"

namespace Global {

    extern Matrix A, B, C;
    extern ifstream fin;
    extern ofstream fout;

    void init(const string &input_file);

    void finalize();

}  // namespace Global