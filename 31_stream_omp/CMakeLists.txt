find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  add_executable(stream_omp stream_omp.cu)
  target_link_libraries(stream_omp PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "OpenMP not found; skipping stream_omp")
endif()
