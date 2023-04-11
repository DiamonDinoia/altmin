#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

MatrixXd matrix_multiplication(MatrixXd n, MatrixXd m) {
    MatrixXd x = n * m;
    return x;
}

int main() {
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2;
    m(0, 1) = -1;
    m(1, 1) = 4;

    MatrixXd n(2, 2);
    n(0, 0)    = 1;
    n(1, 0)    = 1;
    n(0, 1)    = 1;
    n(1, 1)    = 1;

    MatrixXd x = matrix_multiplication(n, m);
    std::cout << x << std::endl;

    std::cout << "hi";
}