#include <CL/opencl.hpp>
#include "blasc.hpp"
#include "deep.hpp"
#include "ocl.hpp"

extern "C" {
  #include "deep.h"
}

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("Matrix padding", "[padding]") {
  ocl_init();

  float inp_arr[] = { 3, 4, 5, 6, };
  float res_arr[] =
    { 0,0,0,0,
      0,3,4,0,
      0,5,6,0,
      0,0,0,0
    };

  Matrix m1 = mat3_of_array(inp_arr, 2, 2, 1);
  Matrix res_mat = mat3_of_array(res_arr, 4, 4, 1);
  Matrix out_mat = mat3_nil(4, 4, 1);

  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  cl::Buffer inp_buf { context, in_flags, mat_mem_size(&m1.matrix), m1.matrix.matrix };
  cl::Buffer out_buf;

  int ret = mat_pad(inp_buf, 1, &m1.matrix, &out_buf);

  REQUIRE(ret == 0);

  ret = queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, mat_mem_size(&out_mat.matrix), out_mat.matrix.matrix);

  std::cout << "Out:\n";
  mat_print(&out_mat.matrix);

  REQUIRE(mat_cmp(res_mat, out_mat));
}

TEST_CASE("Matrix convolution", "[convolution]") {
  ocl_init();

  float inp_arr[] =
    { 10,12,13,16,
      20,21,22,28,
      30,31,35,39,
      40,42,45,50
    };

  float kern_arr[] =
    { 3, 4,
      5, 6,
  };

  float res_arr[] =
    { 304,325,381,
      480,516,587,
      666,713,786,
    };

  Matrix m1 = mat3_of_array(inp_arr, 4, 4, 1);
  Matrix kern = mat3_of_array(kern_arr, 2, 2, 1);

  Matrix res_mat = mat3_of_array(res_arr, 3, 3, 1);
  Matrix out_mat = mat3_nil(3, 3, 1);

  int ret = convolve(&m1.matrix, &kern.matrix, 1, 3, 3, &out_mat.matrix);

  REQUIRE(ret == 0);

  std::cout << "Out:\n";
  mat_print(&out_mat.matrix);

  REQUIRE(mat_cmp(res_mat, out_mat));
}


TEST_CASE("Pooling ff stride 2", "[pooling_2]") {
  ocl_init();

  float inp_arr[] =
    { 10,12,13,16,
      20,21,22,28,
      30,31,35,39,
      40,42,45,50
    };

  float res_arr[] =
    { 21,28,
      42,50,
    };

  Matrix m1 = mat3_of_array(inp_arr, 4, 4, 1);

  Matrix res_mat = mat3_of_array(res_arr, 2, 2, 1);
  Matrix out_mat = mat3_nil(2, 2, 1);

  int ret = pooling_ff(&m1.matrix, 0, 2, 2, 2, 2, 2, &out_mat.matrix);

  REQUIRE(ret == 0);

  std::cout << "Out:\n";
  mat_print(&out_mat.matrix);

  REQUIRE(mat_cmp(res_mat, out_mat));
}


TEST_CASE("Pooling ff", "[pooling]") {
  ocl_init();

  float inp_arr[] =
    { 10,12,13,16,
      20,21,22,28,
      30,31,35,39,
      40,42,45,50
    };

  float res_arr[] =
    { 21,22,28,
      31,35,39,
      42,45,50,
    };

  Matrix m1 = mat3_of_array(inp_arr, 4, 4, 1);

  Matrix res_mat = mat3_of_array(res_arr, 3, 3, 1);
  Matrix out_mat = mat3_nil(3, 3, 1);

  int ret = pooling_ff(&m1.matrix, 0, 1, 3, 3, 2, 2, &out_mat.matrix);

  REQUIRE(ret == 0);

  std::cout << "Out:\n";
  mat_print(&out_mat.matrix);

  REQUIRE(mat_cmp(res_mat, out_mat));
}
