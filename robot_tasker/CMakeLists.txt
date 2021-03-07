cmake_minimum_required(VERSION 3.11)
project(robot_tasker CXX)

set(MSVC 1)
set(MSVC_VERSION 16)
set(CMAKE_CXX_COMPILER_TARGET amd64)
set(CMAKE_CXX_STANDARD 17)

set(EIGEN_OPTIONS -DEIGEN_INITIALIZE_MATRICES_BY_ZERO -DEIGEN_NO_AUTOMATIC_RESIZING)
set(DEBUG_COMPILER_OPTIONS /EHsc /Zi /RTC1 /MDd ${EIGEN_OPTIONS})
set(RELEASE_COMPILER_OPTIONS /W4 /w14640 /EHsc /openmp /O2 /fp:precise /F2048 /arch:AVX2 ${EIGEN_OPTIONS})

add_executable(${PROJECT_NAME} robot_tasker.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")

target_include_directories(${PROJECT_NAME} PRIVATE 
                           X:/Video_Lectures/PGM/libpgm/include/Eigen
                           C:/CppPkgs/vcpkg/installed/x86-windows/include
                           C:/CppPkgs/vcpkg/installed/x64-windows/include
                           C:/CppPkgs/vcpkg/installed/x64-windows/include/LBFGSpp/include
                           C:/CppPkgs/vcpkg/installed/x86-windows/include/cpputil
                           C:/CppPkgs/vcpkg/installed/x86-windows/include/cpputil/cppyplot/include)

target_link_libraries(${PROJECT_NAME} debug 
                      C:/CppPkgs/vcpkg/installed/x64-windows/debug/lib/libzmq-mt-gd-4_3_3.lib
                      C:/CppPkgs/vcpkg/installed/x64-windows/debug/lib/yaml-cpp.lib
                      )

target_link_libraries(${PROJECT_NAME} optimized 
                      C:/CppPkgs/vcpkg/installed/x64-windows/lib/libzmq-mt-4_3_3.lib
                      C:/CppPkgs/vcpkg/installed/x64-windows/lib/yaml-cpp.lib
                      )

# copy runtime dll's
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Debug/libzmq-mt-gd-4_3_3.dll ${CMAKE_BINARY_DIR}/Debug 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Debug/libzmq-mt-gd-4_3_3.pdb ${CMAKE_BINARY_DIR}/Debug 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Debug/yaml-cpp.dll ${CMAKE_BINARY_DIR}/Debug 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Debug/yaml-cpp.pdb ${CMAKE_BINARY_DIR}/Debug 

  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Release/libzmq-mt-4_3_3.dll ${CMAKE_BINARY_DIR}/Release 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Release/libzmq-mt-4_3_3.pdb ${CMAKE_BINARY_DIR}/Release 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Release/yaml-cpp.dll ${CMAKE_BINARY_DIR}/Release 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/../libs/Release/yaml-cpp.pdb ${CMAKE_BINARY_DIR}/Release 
)