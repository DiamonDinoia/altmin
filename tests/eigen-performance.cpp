#include <Eigen/Dense>
#include <chrono>
#include <iostream>

constexpr auto x = 1 << 14;
constexpr auto y = 1 << 10;
//

void colMajor() {
    // allocate a Eigen matrix with 1<<20 rows and 1<<10 columns with random
    // numbers
    Eigen::MatrixXf m1 = Eigen::MatrixXf::Random(x, y);
    Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(y, x);
    // start the timer
    auto start = std::chrono::steady_clock::now();
    // perform the matrix multiplication
    Eigen::MatrixXf m3 = m1 * m2;
    // stop the timer
    auto end = std::chrono::steady_clock::now();
    // print the time difference
    std::cout << "Col major: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    // print m3 mean
    std::cout << "m3.mean() = " << m3.mean() << std::endl;
}

void rowMajor() {
    // allocate a Eigen matrix with 1<<20 rows and 1<<10 columns with random
    // numbers
    Eigen::MatrixXf m1 =
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Random(x, y);
    Eigen::MatrixXf m2 =
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Random(y, x);
    // start the timer
    auto start = std::chrono::steady_clock::now();
    // perform the matrix multiplication
    Eigen::MatrixXf m3 = m1 * m2;
    // stop the timer
    auto end = std::chrono::steady_clock::now();
    // print the time difference
    std::cout << "Row major: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    // print m3 mean
    std::cout << "m3.mean() = " << m3.mean() << std::endl;
}

// main function
int main() {
    colMajor();
    rowMajor();
    return 0;
}
