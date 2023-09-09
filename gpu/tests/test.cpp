#include <CL/opencl.hpp>
#include "../blasc.hpp"
#include "../deep.hpp"
#include "../ocl.hpp"

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

// int main() {
  // ocl_init();
  // return 1;
// }
