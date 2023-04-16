#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

Eigen::MatrixXf lin(const nb::DRef<Eigen::MatrixXf> &input,
                    const nb::DRef<Eigen::MatrixXf> &weight,
                    const nb::DRef<Eigen::MatrixXf> &bias);
Eigen::MatrixXf ReLU(const nb::DRef<Eigen::MatrixXf> &input);
Eigen::MatrixXf sigmoid(const nb::DRef<Eigen::MatrixXf> &input);