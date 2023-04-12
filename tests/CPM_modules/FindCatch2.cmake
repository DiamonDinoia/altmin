include("/home/tom/Documents/Thesis/c++/tests/cmake/CPM_0.38.1.cmake")
CPMAddPackage("NAME;Catch2;GIT_REPOSITORY;https://github.com/catchorg/Catch2.git;GIT_TAG;EXCLUDE_FROM_ALL;YES;GIT_SHALLOW;YES")
set(Catch2_FOUND TRUE)