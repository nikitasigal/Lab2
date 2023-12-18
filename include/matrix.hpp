#pragma once

#include <iostream>

using namespace std;

class Matrix {
public:
    double *data;
    int rows = 3;
    int cols = 3;

    Matrix(int rows, int cols);

    explicit Matrix(int size);

    Matrix();

    Matrix(const Matrix &other);

    Matrix(Matrix &&other) noexcept;

    Matrix &operator=(const Matrix &other);

    Matrix &operator=(Matrix &&other) noexcept;

    ~Matrix();

    double &operator()(int row, int col) const;

    [[nodiscard]] double &T(int row, int col) const;

    [[nodiscard]] Matrix transposed() const;

    [[nodiscard]] int size() const;

    friend Matrix operator*(const Matrix &lhs, const Matrix &rhs);

    friend ostream &operator<<(ostream &out, const Matrix &obj);

    friend istream &operator>>(istream &in, Matrix &obj);
};
