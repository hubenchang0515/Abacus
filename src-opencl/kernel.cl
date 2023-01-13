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