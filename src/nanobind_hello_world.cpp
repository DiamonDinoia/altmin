#include <nanobind/nanobind.h>
#include <iostream>

namespace nb = nanobind;

using namespace nb::literals;

int hello_world(){
    std::cout << "c++: Hello World";
    return 0;
}

NB_MODULE(nanobind_hello_world, m){
    m.def("hello_world", &hello_world);
    m.doc() = "Simple hello_world example";
}


