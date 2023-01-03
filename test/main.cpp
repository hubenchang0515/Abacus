#include "Matrix.h"

int main()
{
    abacus::Matrix mat1(1024, 1024);
    mat1 = 4.0f;

    abacus::Matrix mat2(1024, 1024);
    mat2 = 2.0f;

    // mat1 * mat2;
    std::cout << mat1 * mat2;
}