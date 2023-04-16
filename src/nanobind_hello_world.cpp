#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

using namespace nb::literals;

int hello_world() {
    std::cout << "c++: Hello World";
    return 0;
}

std::string hello_world_out() { return "Hi python from c++"; }

int         hello_world_in(const std::string &av) {
    std::cout << av;
    return 0;
}

NB_MODULE(nanobind_hello_world, m) {
    m.def("hello_world", &hello_world);
    m.def("hello_world_in", &hello_world_in);
    m.def("hello_world_out", &hello_world_out);
    m.doc() = "Simple hello_world example";
}
