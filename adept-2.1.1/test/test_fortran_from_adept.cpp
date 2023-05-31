#include <iostream>
#include <adept_fortran.h>

using namespace adept;

// A subroutine defined in the Fortran file "adept_function.f90"
extern "C" void adept_function(FortranArray* imat,
			       FortranArray* dmat);

int
main()
{
  intMatrix imat = {{2,3,5},{7,11,13}};
  Matrix    dmat = {{2.0, 3.0, 5.0}, {7.0, 11.0, 13.0}};
  dmat = dmat + 0.1;

  std::cout << "######## Arrays in C++ before ########\n";
  std::cout << "  imat = " << imat << "\n";
  std::cout << "  dmat = " << dmat << std::endl;

  adept_function(FortranArray(imat),
		 FortranArray(dmat));

  std::cout << "######## Arrays in C++ after  ########\n";
  std::cout << "  imat = " << imat << "\n";
  std::cout << "  dmat = " << dmat << "\n";

  return 0;
}
