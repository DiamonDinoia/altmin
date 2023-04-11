#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <iostream>
#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

using namespace Eigen;

//using String = std::string;

MatrixXd return_matrix(){
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2;
    m(0,1) = -1;
    m(1,1) = 4;
    return m;
}

NB_MODULE(nanobind_return_matrix, m){
    m.def("return_matrix", &return_matrix);
    m.doc() = "return matrix in c++";
}

