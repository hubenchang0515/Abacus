#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "Accelerator.h"

#include "Matrix.h"

namespace abacus
{

static const std::string source = R"(
__kernel void matFill(__global float* mat, unsigned len, float value)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    mat[idx] = value;
}

__kernel void scalarAddMat(float scalar, __global const float* mat, unsigned len, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = scalar + mat[idx];
}

__kernel void scalarSubMat(float scalar, __global const float* mat, unsigned len, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = scalar - mat[idx];
}

__kernel void scalarMulMat(float scalar, __global const float* mat, unsigned len, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = scalar * mat[idx];
}

__kernel void scalarDivMat(float scalar, __global const float* mat, unsigned len, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = scalar / mat[idx];
}

__kernel void matAddScalar(__global const float* mat, unsigned len, float scalar, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = mat[idx] + scalar;
}

__kernel void matSubScalar(__global const float* mat, unsigned len, float scalar, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = mat[idx] - scalar;
}

__kernel void matMulScalar(__global const float* mat, unsigned len, float scalar, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = mat[idx] * scalar;
}

__kernel void matDivScalar(__global const float* mat, unsigned len, float scalar, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = mat[idx] / scalar;
}

__kernel void matAddMat(__global const float* left, __global const float* right, __global float* result, unsigned len)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = left[idx] + right[idx];
}

__kernel void matSubMat(__global const float* left, __global const float* right, __global float* result, unsigned len)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = left[idx] - right[idx];
}

__kernel void matScalarMulMat(__global const float* left, __global const float* right, __global float* result, unsigned len)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = left[idx] * right[idx];
}

__kernel void matScalarDivMat(__global const float* left, __global const float* right, __global float* result, unsigned len)
{
    unsigned idx = get_global_id(0);

    if (idx >= len)
        return;

    result[idx] = left[idx] / right[idx];
}

__kernel void matMulMat(__global const float* left, unsigned leftWidth, __global const float* right, __global float* result, unsigned width, unsigned height)
{
    unsigned idx = get_global_id(0);

    if (idx >= width * height)
        return;

    unsigned row = idx / width;
    unsigned col = idx % width;

    result[idx] = 0;
    unsigned leftBase = row * leftWidth;
    for (size_t i = 0; i < leftWidth; i++)
    {
        result[idx] += left[leftBase + i] * right[i * width + col];
    }
}

__kernel void matTranspose(__global const float* mat, unsigned width, unsigned height, __global float* result)
{
    unsigned idx = get_global_id(0);

    if (idx >= width * height)
        return;

    unsigned row = idx / width;
    unsigned col = idx % width;

    result[col * height + row] = mat[row * width + col];
}
)";

static cl::Device getFirstGpu()
{
    std::vector<cl::Platform> platforms = Accelerator::platforms();
    if (platforms.size() < 1)
        throw std::runtime_error("OpenCL platform not found, may not install driver.");

    std::vector<cl::Device> devices = Accelerator::devices(platforms[0], CL_DEVICE_TYPE_GPU);
    if (devices.size() < 1)
        throw std::runtime_error("OpenCL GPU device not found, may not install driver.");

    return devices[0];
}

static Accelerator& accelerator()
{
    static Accelerator accel{getFirstGpu(), source};
    return accel;
}
Matrix::~Matrix() noexcept
{
    if (m_buffer != nullptr)
        free(m_buffer);

    m_width = 0;
    m_height = 0;
    m_buffer = nullptr;
}

Matrix::Matrix(const Matrix& src) noexcept:
    m_width{src.m_width},
    m_height{src.m_height},
    m_alignedWidth{m_width},
    m_buffer{static_cast<float*>(malloc(bytes()))}
{
    memcpy(m_buffer, src.m_buffer, bytes());
}

Matrix::Matrix(Matrix&& src) noexcept:
    m_width{src.m_width},
    m_height{src.m_height},
    m_alignedWidth{m_width},
    m_buffer{src.m_buffer}
{
    src.m_width = 0;
    src.m_height = 0;
    src.m_buffer = nullptr;
}


Matrix::Matrix(size_t width, size_t height) noexcept:
    m_width{width},
    m_height{height},
    m_alignedWidth{m_width},
    m_buffer{static_cast<float*>(malloc(bytes()))}
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
        free(m_buffer);

    m_width = src.m_width;
    m_height = src.m_height;
    m_buffer = static_cast<float*>(malloc(bytes()));

    return *this;
}

Matrix& Matrix::operator = (Matrix&& src) noexcept
{
    if (m_buffer != nullptr)
        free(m_buffer);

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
    auto queue = accelerator().queue();

    cl::Buffer mat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(mat, CL_FALSE, 0, bytes(), m_buffer);

    auto op = accelerator().kernel("matAddScalar");
    op.setArg(0, mat);
    op.setArg(1, size());
    op.setArg(2, scalar);
    op.setArg(3, mat);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));

    queue.enqueueReadBuffer(mat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}

Matrix& Matrix::operator -= (float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer mat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(mat, CL_FALSE, 0, bytes(), m_buffer);

    auto op = accelerator().kernel("matSubScalar");
    op.setArg(0, mat);
    op.setArg(1, size());
    op.setArg(2, scalar);
    op.setArg(3, mat);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(mat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}

Matrix& Matrix::operator *= (float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer mat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(mat, CL_FALSE, 0, bytes(), m_buffer);

    auto op = accelerator().kernel("matMulScalar");
    op.setArg(0, mat);
    op.setArg(1, size());
    op.setArg(2, scalar);
    op.setArg(3, mat);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(mat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}

Matrix& Matrix::operator /= (float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer mat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(mat, CL_FALSE, 0, bytes(), m_buffer);

    auto op = accelerator().kernel("matDivScalar");
    op.setArg(0, mat);
    op.setArg(1, size());
    op.setArg(2, scalar);
    op.setArg(3, mat);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(mat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}


Matrix& Matrix::operator += (const Matrix& right)
{
    if (m_width != right.m_width || m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, bytes(), m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matAddMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}

Matrix& Matrix::operator -= (const Matrix& right)
{
    if (m_width != right.m_width || m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, bytes(), m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matSubMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}


Matrix& Matrix::scalarMul (const Matrix& right)
{
    if (m_width != right.m_width || m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, bytes(), m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matScalarMulMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, bytes(), m_buffer);
    
    return *this;
}

Matrix& Matrix::scalarDiv (const Matrix& right)
{
    if (m_width != right.m_width || m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, bytes(), m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matScalarDivMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(size()), accelerator().localSize(size()));
    
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, bytes(), m_buffer);
    
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
    return m_alignedWidth * m_height;
}

size_t Matrix::bytes() const noexcept
{
    return m_alignedWidth * m_height * sizeof(float);
}

Matrix Matrix::transpose() const noexcept
{
    Matrix result(m_height, m_width);
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
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("scalarAddMat");
    op.setArg(0, scalar);
    op.setArg(1, buf);
    op.setArg(2, mat.size());
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator - (float scalar, const Matrix& mat) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("scalarSubMat");
    op.setArg(0, scalar);
    op.setArg(1, buf);
    op.setArg(2, mat.size());
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator * (float scalar, const Matrix& mat) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("scalarMulMat");
    op.setArg(0, scalar);
    op.setArg(1, buf);
    op.setArg(2, mat.size());
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator / (float scalar, const Matrix& mat) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("scalarDivMat");
    op.setArg(0, scalar);
    op.setArg(1, buf);
    op.setArg(2, mat.size());
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}


Matrix operator + (const Matrix& mat, float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("matAddScalar");
    op.setArg(0, buf);
    op.setArg(1, mat.size());
    op.setArg(2, scalar);
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator - (const Matrix& mat, float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("matSubScalar");
    op.setArg(0, buf);
    op.setArg(1, mat.size());
    op.setArg(2, scalar);
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator * (const Matrix& mat, float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("matMulScalar");
    op.setArg(0, buf);
    op.setArg(1, mat.size());
    op.setArg(2, scalar);
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator / (const Matrix& mat, float scalar) noexcept
{
    auto queue = accelerator().queue();

    cl::Buffer buf{accelerator().context(), CL_MEM_READ_WRITE, mat.bytes()};
    queue.enqueueWriteBuffer(buf, CL_FALSE, 0, mat.bytes(), mat.m_buffer);

    auto op = accelerator().kernel("matDivScalar");
    op.setArg(0, buf);
    op.setArg(1, mat.size());
    op.setArg(2, scalar);
    op.setArg(3, buf);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(mat.size()), accelerator().localSize(mat.size()));

    Matrix result(mat.m_width, mat.m_height);
    queue.enqueueReadBuffer(buf, CL_TRUE, 0, result.bytes(), result.m_buffer);
    
    return result;
}

Matrix operator + (const Matrix& left, const Matrix& right)
{
    if (left.m_width != right.m_width || left.m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, left.bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, left.bytes(), left.m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, right.bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matAddMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, left.size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(left.size()), accelerator().localSize(left.size()));
    
    Matrix result(left.m_width, left.m_height);
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, result.bytes(), result.m_buffer);

    return result;
}

Matrix operator - (const Matrix& left, const Matrix& right)
{
    if (left.m_width != right.m_width || left.m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, left.bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, left.bytes(), left.m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, right.bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matSubMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, left.size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(left.size()), accelerator().localSize(left.size()));
    
    Matrix result(left.m_width, left.m_height);
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, result.bytes(), result.m_buffer);
    return result;
}

Matrix scalarMul (const Matrix& left, const Matrix& right)
{
    if (left.m_width != right.m_width || left.m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, left.bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, left.bytes(), left.m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, right.bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matScalarMulMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, left.size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(left.size()), accelerator().localSize(left.size()));
    
    Matrix result(left.m_width, left.m_height);
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, result.bytes(), result.m_buffer);

    return result;
}

Matrix scalarDiv (const Matrix& left, const Matrix& right)
{
    if (left.m_width != right.m_width || left.m_height != right.m_height)
        throw std::range_error("left.m_width != right.m_width || left.m_height != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, left.bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, left.bytes(), left.m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, right.bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);

    auto op = accelerator().kernel("matScalarDivMat");
    op.setArg(0, leftMat);
    op.setArg(1, rightMat);
    op.setArg(2, leftMat);
    op.setArg(3, left.size());
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(left.size()), accelerator().localSize(left.size()));
    
    Matrix result(left.m_width, left.m_height);
    queue.enqueueReadBuffer(leftMat, CL_TRUE, 0, result.bytes(), result.m_buffer);

    return result;
}

Matrix operator * (const Matrix& left, const Matrix& right)
{
    if (left.m_width != right.m_height)
        throw std::range_error("left.m_width != right.m_height");

    auto queue = accelerator().queue();

    cl::Buffer leftMat{accelerator().context(), CL_MEM_READ_WRITE, left.bytes()};
    queue.enqueueWriteBuffer(leftMat, CL_FALSE, 0, left.bytes(), left.m_buffer);
    cl::Buffer rightMat{accelerator().context(), CL_MEM_READ_WRITE, right.bytes()};
    queue.enqueueWriteBuffer(rightMat, CL_FALSE, 0, right.bytes(), right.m_buffer);
    Matrix result(right.m_width, left.m_height);
    cl::Buffer resultMat{accelerator().context(), CL_MEM_READ_WRITE, result.bytes()};

    auto op = accelerator().kernel("matMulMat");
    op.setArg(0, leftMat);
    op.setArg(1, left.m_width);
    op.setArg(2, rightMat);
    op.setArg(3, resultMat);
    op.setArg(4, result.m_width);
    op.setArg(5, result.m_height);
    queue.enqueueNDRangeKernel(op, 0, accelerator().globalSize(result.size()), accelerator().localSize(result.size()));

    queue.enqueueReadBuffer(resultMat, CL_TRUE, 0, result.bytes(), result.m_buffer);

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