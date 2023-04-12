#include <catch2/catch_test_macros.hpp>

#include "../src/nanobind_hello_world.cpp"
#include "../src/nanobind_hello_world_in.cpp"
#include "../src/nanobind_hello_world_out.cpp"
#include "../src/nanobind_matrix_in.cpp"
#include "../src/nanobind_matrix_multiplication.cpp"
#include "../src/nanobind_return_matrix.cpp"

TEST_CASE("Hello world out is correct", "[hello_world_out]") {
    REQUIRE(hello_world_out() == "Hi python from c++");
}
