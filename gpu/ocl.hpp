#include <CL/opencl.hpp>

extern cl::Context context;
extern cl::CommandQueue queue;

extern cl::Program math_prog;
extern cl::Program nn_prog;

#define align(x, al) (((x/al)*al) + ((x % al) ? al : 0)) 
