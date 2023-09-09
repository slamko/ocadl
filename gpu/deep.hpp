#ifndef DEEP_HPP
#define DEEP_HPP

#include <CL/opencl.hpp>

int mat_pad(cl::Buffer &mat_buf, unsigned long padding, const struct mat *mat, cl::Buffer *padded_buf);

#endif

