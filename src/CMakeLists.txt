
add_subdirectory(mm)

# It should depend on generated mm.hpp
add_executable(mm_compare 
    main.cpp)

target_link_libraries(mm_compare mm)


