#include <iostream>
#include "ocl.hpp"
#include <string>
extern "C" {
#include "blasc.h"
#include "ocl.h"
}

int main(int argc, char **argv) {
  using namespace std::chrono;

  std::srand(std::time(nullptr));

  int ret = ocl_init();

  if (argc != 3) {
    std::cerr << "Args: dim kdim\n";
    std::exit(1);
  }

  char *kdim_str = argv[2];
  char *dim_str = argv[1];

  int dim = atoi(dim_str);
  int kdim = atoi(kdim_str);

  struct mat m1 = mat3_random(dim, dim, 1);
  struct mat kernel = mat3_random(kdim, kdim, 1);
  struct mat res1 = {0};
  struct mat res2 = {0};

  // mat_print(&m1);
  auto start = high_resolution_clock::now();
  // conv2(&m1, &kernel, 0, dim - kdim + 1, dim - kdim + 1, &res1);
  convolve(&m1, &kernel, 0, dim - kdim + 1, dim - kdim + 1, &res2);
  auto stop = high_resolution_clock::now();
  auto dur = duration_cast<microseconds>(stop - start);
  std::cout << "Time: " << dur.count() << std::endl;

  /*
  printf("Innov\n");
  mat_print(&res1);
  printf("Trad\n");
  mat_print(&res2);
  */

  ocl_finish();
}
