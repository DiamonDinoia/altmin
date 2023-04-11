#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

using namespace nb::literals;

// using String = std::string;

std::string hello_world_out() { return "Hi python from c++"; }

NB_MODULE(nanobind_hello_world_out, m) {
    m.def("hello_world_out", &hello_world_out);
    m.doc() = "Simple hello_world example where it outputs to python";
}
