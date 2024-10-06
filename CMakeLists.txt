cmake_minimum_required(VERSION 3.17)
project(YourProjectName CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add your CUDA source file
add_executable(run src/main.cu)

# Set CUDA architecture (change according to your GPU)
set_property(TARGET run PROPERTY CUDA_ARCHITECTURES 61)

# Link cuBLAS
target_link_libraries(run PRIVATE cublas)

# Set other necessary flags