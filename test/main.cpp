#include "Matrix.h"

int main()
{
    abacus::Matrix mat1(13, 3);
    mat1 = 4.0f;

    abacus::Matrix mat2(13, 3);
    mat2 = 2.0f;

    std::cout << (mat1 += 2.0);
    std::cout << (mat1 -= 2.0);
    std::cout << (mat1 *= 2.0);
    std::cout << (mat1 /= 2.0);

    std::cout << (mat1 += mat2);
    std::cout << (mat1 -= mat2);
    std::cout << (mat1.scalarMul(mat2));
    std::cout << (mat1.scalarDiv(mat2));

    std::cout << mat1 + 2.0;
    std::cout << mat1 - 2.0;
    std::cout << mat1 * 2.0;
    std::cout << mat1 / 2.0;

    std::cout << 2.0 + mat1;
    std::cout << 2.0 - mat1;
    std::cout << 2.0 * mat1;
    std::cout << 2.0 / mat1;

    std::cout << mat1 + mat2;
    std::cout << mat1 - mat2;
    std::cout << abacus::scalarMul(mat1, mat2);
    std::cout << abacus::scalarDiv(mat1, mat2);

    std::cout << mat1 * mat2.transpose();
}