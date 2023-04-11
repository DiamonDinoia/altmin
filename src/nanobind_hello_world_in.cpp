#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

using namespace nb::literals;

// using String = std::string;

int hello_world_in(const std::string &av) {
    std::cout << av;
    return 0;
}

NB_MODULE(nanobind_hello_world_in, m) {
    m.def("hello_world_in", &hello_world_in);
    m.doc() = "Simple hello_world example where it takes input from python";
}
