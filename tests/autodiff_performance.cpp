#include <torch/torch.h>

#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <chrono>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>

#include "engine.hpp"

void libtorch_simple() {
    torch::Tensor x     = torch::rand({500, 1000}).requires_grad_(true);
    torch::Tensor y     = torch::rand({500, 1000}).requires_grad_(true);
    auto          start = std::chrono::steady_clock::now();
    torch::Tensor z     = (x + y).mean();
    z.backward();
    torch::Tensor x_grad = x.grad();
    torch::Tensor y_grad = y.grad();
    auto          end    = std::chrono::steady_clock::now();
    std::cout << "libtorch: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    // std::cout << x_grad << std::endl;
}

// Only works for scalar so not relevant
void ugrad_simple() {
    auto a = ugrad::make_shared<ugrad::Value>(-4.0f);
    auto b = ugrad::make_shared<ugrad::Value>(2.0f);
    auto c = a + b;
    auto d = a * b + b * b * b;
    c      = c + c + 1;
    c      = c + 1 + c + (-a);
    d      = d + d * 2 + (b + a)->relu();
    d      = d + 3 * d + (b - a)->relu();
    auto e = c - d;
    auto f = e * e;
    auto g = f / 2.0;
    g      = g + 10.0 / f;
    std::cout << *g << std::endl;  //   Res of forward pass
    g->backward();
    std::cout << *a << std::endl;  //   dg/da
    std::cout << *b << std::endl;  //   dg/db
}

// Doesn't work and slow anyway
//  void diffNum_simple() {
//      using dmath = Math<ddouble<0>>;
//      // Example 1. a, b are variables. c = a+b; d = log(max(sin(a/c), b))
//      ddouble<0> a = 2., b = 3.;
//      // 2 total variables, a is the first, b is the second
//      a.setVar(2, 0);
//      b.setVar(2, 1);

//     auto c = a + b;
//     auto d = dmathd::Log(dmath::Max(dmathd::Sin(a / c), b));

//     std::cout << d << std::endl;
// }

// autodiff::MatrixXreal f(const autodiff::MatrixXreal &x,
//                         const autodiff::MatrixXreal &y) {
//     return x + y;
// }

// void autodiff_simple() {
//     Eigen::MatrixXd       x          = Eigen::MatrixXd::Random(500, 1000);
//     Eigen::MatrixXd       y          = Eigen::MatrixXd::Random(500, 1000);
//     autodiff::MatrixXreal x_autodiff = x;
//     autodiff::MatrixXreal y_autodiff = x;

//     autodiff::MatrixXreal z;

//     Eigen::MatrixXd       x_grad =
//         autodiff::jacobian(f, autodiff::wrt(x), autodiff::at(x), z);
//     // Eigen::MatrixXd y_grad =
//     //     autodiff::jacobian(f, autodiff::wrt(y), autodiff::at(x, y));

//     std::cout << x_grad << std::endl;
// }

// ugrad only works on scalar values so no use

// diffNum The time complexity is greatly many times of reversed differentiating
// algorithms (back propagation) when there is large number of independent
// variables. Thus it can be extremely inefficient when there are many
// variables! but maybe I still try but not a priority

// autodiff dual both ways
// real is forward
// var is backwards
// I don't know how it works tho

// Boost forward model - only works with scalars - too complicated imo

// Last one is actually probably good but I have no idea how to use it
// And no cmakelist so very hard to link

// Eigen unsupported autodiff is too complicated.

int main() {
    libtorch_simple();
    ugrad_simple();
    // autodiff_simple();
    return 0;
}