#include "matrix.hpp"

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(new double[rows * cols]) {
    fill(data, data + rows * cols, 0);
}

Matrix::Matrix(int size) : Matrix(size, size) {}

Matrix::Matrix() : Matrix(3, 3) {}

Matrix::Matrix(const Matrix &other) : rows(other.rows),
                                      cols(other.cols),
                                      data(new double[other.rows * other.cols]) {
    copy(other.data, other.data + rows * cols, data);
}

Matrix::Matrix(Matrix &&other) noexcept: rows(other.rows), cols(other.cols), data(other.data) {
    other.data = nullptr;
}

Matrix &Matrix::operator=(const Matrix &other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = new double[rows * cols];
        copy(other.data, other.data + rows * cols, data);
    }
    return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;

        other.data = nullptr;
    }
    return *this;
}

Matrix::~Matrix() {
    delete[] data;
}

double &Matrix::operator()(int row, int col) const {
    return data[row * cols + col];
}

double &Matrix::T(int row, int col) const {
    return (*this)(col, row);
}

Matrix Matrix::transposed() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(j, i) = (*this)(i, j);
    return result;
}

int Matrix::size() const {
    return rows * cols;
}

Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.cols != rhs.rows) {
        cerr << "Cannot multiply matrix of size (" << lhs.rows << ',' << lhs.cols << ") by matrix of size (" << rhs.rows
             << ',' << rhs.cols << ")\n";
        exit(1);
    }
    Matrix result = Matrix(lhs.rows, rhs.cols);
    for (int k = 0; k < result.cols; k++)
        for (int i = 0; i < lhs.rows; i++)
            for (int j = 0; j < lhs.cols; j++)
                result(i, k) += lhs(i, j) * rhs(j, k);
    return result;
}

ostream &operator<<(ostream &out, const Matrix &obj) {
    for (int i = 0; i < obj.rows; i++) {
        for (int j = 0; j < obj.cols; j++)
            out << obj(i, j) << ' ';
        out << '\n';
    }
    return out;
}

istream &operator>>(istream &in, Matrix &obj) {
    for (int i = 0; i < obj.rows; i++) {
        for (int j = 0; j < obj.cols; j++)
            in >> obj(i, j);
    }
    return in;
}