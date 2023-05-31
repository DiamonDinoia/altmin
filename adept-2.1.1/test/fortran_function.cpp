#include <iostream>
#include <adept_fortran.h>

using namespace adept;

// A subroutine defined in the Fortran file "adept_function.f90"
extern "C"
void fortran_function(FortranArray* im, FortranArray* dm) {
  intMatrix imat;
  Matrix    dmat;

  imat >>= im;
  dmat >>= dm;

  std::cout << "######## Arrays in C++ ########\n";
  std::cout << "  imat = " << imat << "\n";
  std::cout << "  dmat = " << dmat << "\n";

  try {
    std::cout << "Trying to associate Fortran array with wrong type of Adept array\n";
    intVector ivec;
    ivec >>= dm;
  }
  catch (const adept::exception& ex) {
    std::cout << "ERROR CAUGHT: " << ex.what() << "\n";
  }
  
  
  std::cout << "Multiplying arrays by 2 in C++" << std::endl;
  imat *= 2;
  dmat *= 2.0;
}
