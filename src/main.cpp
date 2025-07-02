// #include <cuda_runtime.h>
// #include <limits.h>
#include <pybind11/pybind11.h>
// #include <unistd.h>
// #include <iostream>
// #include <string>
// #include <thread>

std::string hello_from_bin() { return "Hello from bes-velocimetry!"; }

// std::string hello_from_bin() {
//   int deviceCount = 0;
//   int rank, nprocs;
//   cudaGetDeviceCount(&deviceCount);

//   char hostname[HOST_NAME_MAX];
//   gethostname(hostname, HOST_NAME_MAX);
//   printf("%s: Rank %d out of %d processes: I see %d GPU(s)\n", hostname, rank, nprocs, deviceCount);

//   int dev, len = 15;
//   char gpu_id[15];
//   cudaDeviceProp deviceProp;

//   for (dev = 0; dev < deviceCount; ++dev) {
//     cudaSetDevice(dev);
//     cudaGetDeviceProperties(&deviceProp, dev);
//     cudaDeviceGetPCIBusId(gpu_id, len, dev);
//     printf("%d for rank %d: %s\n", dev, rank, gpu_id);
//   }

//   printf("%s: done!\n", hostname);
//   return std::to_string(deviceCount);
// }

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "pybind11 hello module";

  m.def("hello_from_bin", &hello_from_bin, R"pbdoc(
      A function that returns a Hello string.
  )pbdoc");
};
