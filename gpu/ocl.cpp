#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

extern "C" {
#include "ocl.h"
}

cl::Context context;
cl::CommandQueue queue;

cl::Program math_prog;
cl::Program nn_prog;
cl::Device device;

int get_default_device(cl::Device *device) {
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
      *device = devices.front();
      return 0;
    }
  }
    
  *device = NULL;
  std::cerr << "No OpenCL supported devices found\n";
  return 1;
}

int build_prog(std::string source_name, cl::Device device, cl::Program *prog) {
  std::ifstream prog_file { source_name };

  if (!prog_file.is_open()) {
    std::cerr << "No source file found\n";
    return 1;
  }

  std::stringstream buf;
  buf << prog_file.rdbuf();
  std::string src = buf.str();

  cl::Program::Sources sources;
  sources.push_back({src.c_str(), src.length()});

  *prog = cl::Program (context, sources);

  if (prog->build({device}) != CL_SUCCESS) {
    std::cerr << "CL build error: " << prog->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    return 1;
  }

  return 0;
}

extern "C" int ocl_init() {
  using namespace cl;

  int ret = 1;

  if ((ret = get_default_device(&device))) {
    return ret;
  }
  
  context = Context(device);
  queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

  if ((ret = build_prog("gpu/nn.cl", device, &nn_prog))) {
    std::cerr << "NN lib build failed\n";
    return ret;
  }

  if ((ret = build_prog("gpu/add.cl", device, &math_prog))) {
    std::cerr << "Math lib build failed\n";
    return ret;
  }

  return 0;
}

extern "C" int ocl_finish() {
  return queue.finish();
}
