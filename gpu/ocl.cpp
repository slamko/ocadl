#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>

extern "C" {
#include "ocl.h"
}

cl::Context context;
cl::CommandQueue queue;

cl::Program math_prog;
cl::Program nn_prog;

int get_default_device(cl::Device &device) {
  using namespace cl;
  std::vector<Platform> platforms;

  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    std::cerr << "No OpenCL platforms found\n";
    return 1;
  }
  
  std::vector<Device> devices;
  for (Platform platform : platforms) {
    
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (!devices.empty()) {
      device = devices.front();
      return 0;
    }
  }
    
  std::cerr << "No OpenCL supported devices found\n";
  return 1;
}

int build_prog(std::string source_name, cl::Program &prog) {
  std::ifstream prog_file { source_name };

  if (!prog_file.is_open()) {
    std::cerr << "No source file found\n";
    return 1;
  }

  std::string src;
  prog_file >> src;

  cl::Program::Sources sources {src};
  prog = cl::Program (context, sources);

  return 0;
}

extern "C" int ocl_init() {
  using namespace cl;

  int ret = 1;
  Device device;

  if ((ret = get_default_device(device))) {
    return ret;
  }
  
  context = Context(device);
  queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

  if ((ret = build_prog("gpu/add.cl", math_prog))) {
    return ret;
  }

  if ((ret = build_prog("gpu/nn.cl", nn_prog))) {
    return ret;
  }

  return 0;
}
