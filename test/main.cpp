#include "Matrix.h"
#include <sys/time.h>

#define N 100

int main()
{
    abacus::Matrix mat1(13, 3);
    mat1 = 4.0f;

    abacus::Matrix mat2(13, 3);
    mat2 = 2.0f;

    std::cout << mat1 + mat2;
    std::cout << mat1 - mat2;
    std::cout << abacus::scalarMul(mat1, mat2);
    std::cout << abacus::scalarDiv(mat1, mat2);
    std::cout << mat1 * mat2.transpose();
}