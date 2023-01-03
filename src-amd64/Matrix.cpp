#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
    #include <malloc.h>
#endif

#if defined(__GNUC__)
    #include <x86intrin.h>
    #define ALIGN(X) __attribute__((aligned(X)))
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define ALIGN(X) __declspec(align(X))
#endif

#include "Matrix.h"

namespace abacus
{

constexpr static const size_t SIMD_BITS = 256;

constexpr static const size_t ALIGN_BYTES = SIMD_BITS / 8;

constexpr static const size_t SIMD_FLOATMM_LEN = (SIMD_BITS / 8 / sizeof(float));

constexpr static size_t SIMD_FLOATMM_LAST(size_t N)
{
    return N / SIMD_FLOATMM_LEN * SIMD_FLOATMM_LEN;
}

constexpr static size_t SIMD_ALIGNED_BYTES(size_t bytes, size_t align)
{
    return (bytes + align - 1) / align * align;
} 

constexpr static size_t SIMD_FLOATMM_ALIGNED_LENGTH(size_t n)
{
    return SIMD_ALIGNED_BYTES(n * sizeof(float), ALIGN_BYTES) / sizeof(float);
}

static float* simd_floatmm_alloc(size_t bytes)
{
    #ifdef _WIN32
        return static_cast<float*>(_aligned_malloc(bytes, ALIGN_BYTES));
    #elif defined(__unix__)
        return static_cast<float*>(aligned_alloc(ALIGN_BYTES, bytes));
    #endif
}

static void simd_floatmm_free(float* ptr)
{
    #ifdef _WIN32
        return _aligned_free(ptr);
    #elif defined(__unix__)
        return free(ptr);
    #endif
}

Matrix::~Matrix() noexcept
{
    if (m_buffer != nullptr)
        simd_floatmm_free(m_buffer);

    m_width = 0;
    m_height = 0;
    m_buffer = nullptr;
}

Matrix::Matrix(const Matrix& src) noexcept:
    m_width{src.m_width},
    m_height{src.m_height},
    m_alignedWidth{SIMD_FLOATMM_ALIGNED_LENGTH(m_width)},
    m_buffer{simd_floatmm_alloc(bytes())}
{
    memcpy(m_buffer, src.m_buffer, bytes());
}

Matrix::Matrix(Matrix&& src) noexcept:
    m_width{src.m_width},
    m_height{src.m_height},
    m_alignedWidth{SIMD_FLOATMM_ALIGNED_LENGTH(m_width)},
    m_buffer{src.m_buffer}
{
    src.m_width = 0;
    src.m_height = 0;
    src.m_buffer = nullptr;
}


Matrix::Matrix(size_t width, size_t height) noexcept:
    m_width{width},
    m_height{height},
    m_alignedWidth{SIMD_FLOATMM_ALIGNED_LENGTH(m_width)},
    m_buffer{simd_floatmm_alloc(bytes())}
{
    
}

float* Matrix::operator [] (size_t row)
{
    return m_buffer + row * m_alignedWidth;
}

const float* Matrix::operator [] (size_t row) const
{
    return m_buffer + row * m_alignedWidth;
}

Matrix& Matrix::operator = (const Matrix& src) noexcept
{
    if (m_buffer != nullptr)
        simd_floatmm_free(m_buffer);

    m_width = src.m_width;
    m_height = src.m_height;
    m_buffer = simd_floatmm_alloc(bytes());

    return *this;
}

Matrix& Matrix::operator = (Matrix&& src) noexcept
{
    if (m_buffer != nullptr)
        simd_floatmm_free(m_buffer);

    m_width = src.m_width;
    m_height = src.m_height;
    m_buffer = src.m_buffer;

    src.m_width = 0;
    src.m_height = 0;
    src.m_buffer = nullptr;

    return *this;
}

Matrix& Matrix::operator = (float value) noexcept
{
    #pragma omp parallel for
    for (size_t row = 0; row < m_height; row++)
    {
        for (size_t col = 0; col < m_width; col++)
        {
            (*this)[row][col] = value;
        }
    }
    return *this;
}


Matrix& Matrix::operator += (float scalar) noexcept
{
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(m_buffer + i);
        mVec = _mm256_add_ps(mVec, sVec);
        _mm256_store_ps(m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] += scalar;
    }
    
    return *this;
}

Matrix& Matrix::operator -= (float scalar) noexcept
{
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(m_buffer + i);
        mVec = _mm256_sub_ps(mVec, sVec);
        _mm256_store_ps(m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] -= scalar;
    }
    
    return *this;
}

Matrix& Matrix::operator *= (float scalar) noexcept
{
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(m_buffer + i);
        mVec = _mm256_mul_ps(mVec, sVec);
        _mm256_store_ps(m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] *= scalar;
    }
    
    return *this;
}

Matrix& Matrix::operator /= (float scalar) noexcept
{
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(m_buffer + i);
        mVec = _mm256_div_ps(mVec, sVec);
        _mm256_store_ps(m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] /= scalar;
    }
    
    return *this;
}


Matrix& Matrix::operator += (const Matrix& right)
{
    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_add_ps(lVec, rVec);
        _mm256_store_ps(m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] += right.m_buffer[i];
    }
    
    return *this;
}

Matrix& Matrix::operator -= (const Matrix& right)
{
    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_sub_ps(lVec, rVec);
        _mm256_store_ps(m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] -= right.m_buffer[i];
    }
    
    return *this;
}


Matrix& Matrix::scalarMul (const Matrix& right)
{
    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_mul_ps(lVec, rVec);
        _mm256_store_ps(m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] *= right.m_buffer[i];
    }
    
    return *this;
}

Matrix& Matrix::scalarDiv (const Matrix& right)
{
    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_div_ps(lVec, rVec);
        _mm256_store_ps(m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(size()); i < size(); i += 1)
    {
        m_buffer[i] /= right.m_buffer[i];
    }
    
    return *this;
}


size_t Matrix::width() const noexcept
{
    return m_width;
}

size_t Matrix::height() const noexcept
{
    return m_height;
}

size_t Matrix::size() const noexcept
{
    return m_width * m_height;
}

size_t Matrix::bytes() const noexcept
{
    return m_alignedWidth * m_height * sizeof(float);
}

Matrix Matrix::transpose() const noexcept
{
    Matrix result(m_height, m_width);
    #pragma omp parallel for
    for (size_t row = 0; row < result.m_height; row++)
    {
        for (size_t col = 0; col < result.m_width; col++)
        {
            result[row][col] = (*this)[col][row];
        }
    }
    return result;
}


Matrix operator + (float scalar, const Matrix& mat) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_add_ps(sVec, mVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = scalar + mat.m_buffer[i];
    }

    return result;
}

Matrix operator - (float scalar, const Matrix& mat) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_sub_ps(sVec, mVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = scalar - mat.m_buffer[i];
    }

    return result;
}

Matrix operator * (float scalar, const Matrix& mat) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_mul_ps(sVec, mVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = scalar * mat.m_buffer[i];
    }

    return result;
}

Matrix operator / (float scalar, const Matrix& mat) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_div_ps(sVec, mVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = scalar / mat.m_buffer[i];
    }

    return result;
}


Matrix operator + (const Matrix& mat, float scalar) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_add_ps(mVec, sVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = mat.m_buffer[i] + scalar;
    }

    return result;
}

Matrix operator - (const Matrix& mat, float scalar) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_sub_ps(mVec, sVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = mat.m_buffer[i] - scalar;
    }

    return result;
}

Matrix operator * (const Matrix& mat, float scalar) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_mul_ps(mVec, sVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = mat.m_buffer[i] * scalar;
    }

    return result;
}

Matrix operator / (const Matrix& mat, float scalar) noexcept
{
    Matrix result(mat.m_width, mat.m_height);
    __m256 sVec = _mm256_set1_ps(scalar);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(mat.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 mVec = _mm256_load_ps(mat.m_buffer + i);
        mVec = _mm256_div_ps(mVec, sVec);
        _mm256_store_ps(result.m_buffer + i, mVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(mat.size()); i < mat.size(); i += 1)
    {
        result.m_buffer[i] = mat.m_buffer[i] / scalar;
    }

    return result;
}

Matrix operator + (const Matrix& left, const Matrix& right)
{
    Matrix result(left.m_width, left.m_height);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(left.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(left.m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_add_ps(lVec, rVec);
        _mm256_store_ps(result.m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(left.size()); i < left.size(); i += 1)
    {
        result.m_buffer[i] = left.m_buffer[i] + right.m_buffer[i];
    }

    return result;
}

Matrix operator - (const Matrix& left, const Matrix& right)
{
    Matrix result(left.m_width, left.m_height);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(left.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(left.m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_sub_ps(lVec, rVec);
        _mm256_store_ps(result.m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(left.size()); i < left.size(); i += 1)
    {
        result.m_buffer[i] = left.m_buffer[i] - right.m_buffer[i];
    }

    return result;
}

Matrix scalarMul (const Matrix& left, const Matrix& right)
{
    Matrix result(left.m_width, left.m_height);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(left.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(left.m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_mul_ps(lVec, rVec);
        _mm256_store_ps(result.m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(left.size()); i < left.size(); i += 1)
    {
        result.m_buffer[i] = left.m_buffer[i] * right.m_buffer[i];
    }

    return result;
}

Matrix scalarDiv (const Matrix& left, const Matrix& right)
{
    Matrix result(left.m_width, left.m_height);

    #pragma omp parallel for
    for (size_t i = 0; i < SIMD_FLOATMM_LAST(left.size()); i += SIMD_FLOATMM_LEN)
    {
        __m256 lVec = _mm256_load_ps(left.m_buffer + i);
        __m256 rVec = _mm256_load_ps(right.m_buffer + i);
        lVec = _mm256_div_ps(lVec, rVec);
        _mm256_store_ps(result.m_buffer + i, lVec);
    }

    #pragma omp parallel for
    for (size_t i = SIMD_FLOATMM_LAST(left.size()); i < left.size(); i += 1)
    {
        result.m_buffer[i] = left.m_buffer[i] / right.m_buffer[i];
    }

    return result;
}

Matrix operator * (const Matrix& left, const Matrix& right)
{
    auto transpose = right.transpose();
    Matrix result(left.m_width, left.m_height);

    #pragma omp parallel for
    for (size_t row = 0; row < result.m_height; row++)
    {
        for (size_t col = 0; col < result.m_width; col++)
        {
            size_t i = 0;
            __m256 sumVec = _mm256_set1_ps(0.0f);
            for (; i < SIMD_FLOATMM_LAST(left.m_width); i += SIMD_FLOATMM_LEN)
            {
                __m256 lVec = _mm256_load_ps(left.m_buffer + row * left.m_alignedWidth + i);
                __m256 rVec = _mm256_load_ps(transpose.m_buffer + col * transpose.m_alignedWidth + i); // cache miss
                lVec = _mm256_mul_ps(lVec, rVec);
                sumVec = _mm256_add_ps(sumVec, lVec);
            }

            float sum[SIMD_FLOATMM_LEN] ALIGN(ALIGN_BYTES);
            
            _mm256_store_ps(sum, sumVec);

            result[row][col] = 0;
            for (size_t j = 0; j < SIMD_FLOATMM_LEN; j++)
            {
                result[row][col] += sum[j];
            }

            for (; i < left.m_width; i += 1)
            {
                result[row][col] += left[row][i] * transpose[col][i];
            }
        }
    }

    return result;
}

std::ostream& operator<< (std::ostream& stream, const Matrix& mat)
{
    stream << "[\n";
    for (size_t y = 0; y < mat.m_height; y++)
    {
        stream << "  [ ";
        for (size_t x = 0; x < mat.m_width; x++)
        {
            stream << mat[y][x] << " ";

            if (mat.m_width > 6 && x == 2) 
            {
                stream << "... ";
                x = mat.m_width - 4;
            }
        }
        stream << "]\n";
    }
    stream << "]\n";
    return stream;
}

}; // namespace abacus