#ifndef ABACUS_MATRIX_H
#define ABACUS_MATRIX_H

#include <functional>
#include <iostream>

namespace abacus
{

class Matrix
{
    friend Matrix operator + (float scalar, const Matrix& mat) noexcept;
    friend Matrix operator - (float scalar, const Matrix& mat) noexcept;
    friend Matrix operator * (float scalar, const Matrix& mat) noexcept;
    friend Matrix operator / (float scalar, const Matrix& mat) noexcept;

    friend Matrix operator + (const Matrix& mat, float scalar) noexcept;
    friend Matrix operator - (const Matrix& mat, float scalar) noexcept;
    friend Matrix operator * (const Matrix& mat, float scalar) noexcept;
    friend Matrix operator / (const Matrix& mat, float scalar) noexcept;

    friend Matrix operator + (const Matrix& left, const Matrix& right);
    friend Matrix operator - (const Matrix& left, const Matrix& right);
    friend Matrix scalarMul (const Matrix& left, const Matrix& right);
    friend Matrix scalarDiv (const Matrix& left, const Matrix& right);

    friend Matrix operator * (const Matrix& left, const Matrix& right);

    friend std::ostream& operator<< (std::ostream& stream, const Matrix& mat);
public:

    ~Matrix() noexcept;
    Matrix(const Matrix& src) noexcept;
    Matrix(Matrix&& src) noexcept;
    Matrix(size_t width, size_t height) noexcept;

    float* operator [] (size_t row);
    const float* operator [] (size_t row) const;

    Matrix& operator = (const Matrix& src) noexcept;
    Matrix& operator = (Matrix&& src) noexcept;
    Matrix& operator = (float value) noexcept;

    Matrix& operator += (float scalar) noexcept;
    Matrix& operator -= (float scalar) noexcept;
    Matrix& operator *= (float scalar) noexcept;
    Matrix& operator /= (float scalar) noexcept;

    Matrix& operator += (const Matrix& right);
    Matrix& operator -= (const Matrix& right);
    Matrix& scalarMul (const Matrix& right);
    Matrix& scalarDiv (const Matrix& right);

    size_t width() const noexcept;
    size_t height() const noexcept;

    size_t size() const noexcept;
    size_t bytes() const noexcept;

    Matrix transpose() const noexcept;

private:
    size_t m_width;
    size_t m_height;
    size_t m_alignedWidth;
    float* m_buffer;

    Matrix& opScalar (std::function<float(float,float)> func, float scalar) noexcept;
    Matrix& opMat (std::function<float(float,float)> func, const Matrix& right);

}; // class Matrix

// static Matrix opScalar (cl::Kernel op, float scalar, const Matrix& mat) noexcept;
Matrix operator + (float scalar, const Matrix& mat) noexcept;
Matrix operator - (float scalar, const Matrix& mat) noexcept;
Matrix operator * (float scalar, const Matrix& mat) noexcept;
Matrix operator / (float scalar, const Matrix& mat) noexcept;

// static Matrix opScalar (cl::Kernel op, const Matrix& mat, float scalar) noexcept;
Matrix operator + (const Matrix& mat, float scalar) noexcept;
Matrix operator - (const Matrix& mat, float scalar) noexcept;
Matrix operator * (const Matrix& mat, float scalar) noexcept;
Matrix operator / (const Matrix& mat, float scalar) noexcept;

// static Matrix opMat (cl::Kernel op, const Matrix& left, const Matrix& right);
Matrix operator + (const Matrix& left, const Matrix& right);
Matrix operator - (const Matrix& left, const Matrix& right);
Matrix scalarMul (const Matrix& left, const Matrix& right);
Matrix scalarDiv (const Matrix& left, const Matrix& right);

Matrix operator * (const Matrix& left, const Matrix& right);

}; // namespace abacus

#endif // ABACUS_MATRIX_H