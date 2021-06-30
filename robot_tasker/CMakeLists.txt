cmake_minimum_required(VERSION 3.11)
project(robot_tasker CXX)

set(MSVC 1)
set(MSVC_VERSION 16)
set(CMAKE_CXX_COMPILER_TARGET amd64)
set(CMAKE_CXX_STANDARD 17)

set(EIGEN_OPTIONS -DEIGEN_INITIALIZE_MATRICES_BY_ZERO -DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_STACK_ALLOCATION_LIMIT=0)
set(DEBUG_COMPILER_OPTIONS /EHsc /Zi /RTC1 /MDd /INCREMENTAL:NO ${EIGEN_OPTIONS})
set(RELEASE_COMPILER_OPTIONS /W4 /w14640 /EHsc /openmp /O2 /fp:precise /F2048 /arch:AVX2 ${EIGEN_OPTIONS})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/external_libs.cmake)

add_library(util src/util/util.cpp)
target_link_libraries(util EIGEN YAML_CPP)

add_library(to_drive src/learning/to_drive/to_drive_util.cpp
                     src/learning/to_drive/robot_dynamics.cpp)
target_link_libraries(to_drive EIGEN YAML_CPP)

add_executable(${PROJECT_NAME} src/robot_tasker.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")

target_link_libraries(${PROJECT_NAME} 
                      CPPYPLOT
                      util
                      to_drive)

# copy runtime dll's
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Debug/libzmq-mt-gd-4_3_3.dll ${CMAKE_BINARY_DIR}/Debug 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Debug/libzmq-mt-gd-4_3_3.pdb ${CMAKE_BINARY_DIR}/Debug 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Debug/yaml-cpp.dll ${CMAKE_BINARY_DIR}/Debug 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Debug/yaml-cpp.pdb ${CMAKE_BINARY_DIR}/Debug 

  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Release/libzmq-mt-4_3_3.dll ${CMAKE_BINARY_DIR}/Release 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Release/libzmq-mt-4_3_3.pdb ${CMAKE_BINARY_DIR}/Release 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Release/yaml-cpp.dll ${CMAKE_BINARY_DIR}/Release 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Release/yaml-cpp.pdb ${CMAKE_BINARY_DIR}/Release 
)