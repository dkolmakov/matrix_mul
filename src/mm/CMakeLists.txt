
add_library(sum_kernels INTERFACE)
target_include_directories(sum_kernels INTERFACE 
                           "${CMAKE_CURRENT_SOURCE_DIR}/sum_kernels"
                           "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_library(mul_kernels INTERFACE)
target_include_directories(mul_kernels INTERFACE 
                           "${CMAKE_CURRENT_SOURCE_DIR}/mul_kernels"
                           "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_executable(mm_profiler matrix_mul_tests.cpp)
target_link_libraries(mm_profiler avprof sum_kernels mul_kernels)

add_library(mm INTERFACE)
target_include_directories(mm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(mm_profiler_run
  COMMAND mm_profiler 32 100 > mm.hpp 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated mm_profiler in ${CMAKE_CURRENT_SOURCE_DIR}"
)

add_dependencies(mm mm_profiler_run)

