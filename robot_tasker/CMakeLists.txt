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

# Util
add_library(util src/util/util.cpp)
target_compile_options(util PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
target_compile_options(util PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")
target_link_libraries(util EIGEN YAML_CPP)

# Learning to drive
add_library(to_drive src/learning/to_drive/to_drive_util.cpp
                     src/learning/to_drive/robot_dynamics.cpp)
target_compile_options(to_drive PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
target_compile_options(to_drive PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")
target_link_libraries(to_drive EIGEN YAML_CPP)

# Gui
add_library(gui src/gui/gui.cpp 
                src/gui/gui_util.cpp)
target_compile_options(gui PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
target_compile_options(gui PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")
target_link_libraries(gui IMGUI IMGUI_SFML SFML util CPPYPLOT)

# Robot tasker
add_executable(${PROJECT_NAME} src/robot_tasker.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")
target_link_libraries(${PROJECT_NAME} 
                      CPPYPLOT
                      util
                      to_drive
                      gui)

# Benchmarking
# add_executable(ann_runtime src/tests/ann_runtime.cpp)
# target_compile_options(ann_runtime PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_COMPILER_OPTIONS}>")
# target_compile_options(ann_runtime PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_COMPILER_OPTIONS}>")
# target_link_libraries(ann_runtime BENCHMARK EIGEN)

# copy runtime dll's
set(DEBUG_DLLS freetyped.dll
               freetyped.pdb
               libpng16d.dll
               libpng16d.pdb
               libzmq-mt-gd-4_3_3.dll
               libzmq-mt-gd-4_3_3.pdb
               sfml-audio-d.pdb
               sfml-audio-d-2.dll
               sfml-graphics-d.pdb
               sfml-graphics-d-2.dll
               sfml-network-d.pdb
               sfml-network-d-2.dll
               sfml-system-d.pdb
               sfml-system-d-2.dll
               sfml-window-d.pdb
               sfml-window-d-2.dll
               yaml-cpp.dll
               yaml-cpp.pdb
               zlibd.pdb
               zlibd1.dll
               bz2d.dll
               bz2d.pdb)

set(RELEASE_DLLS sfml-network.pdb
                 sfml-network-2.dll
                 sfml-system.pdb
                 sfml-system-2.dll
                 sfml-window.pdb
                 sfml-window-2.dll
                 yaml-cpp.dll
                 yaml-cpp.pdb
                 zlib.pdb
                 zlib1.dll
                 bz2.dll
                 bz2.pdb
                 freetype.dll
                 freetype.pdb
                 libpng16.dll
                 libpng16.pdb
                 libzmq-mt-4_3_3.dll
                 libzmq-mt-4_3_3.pdb
                 sfml-audio.pdb
                 sfml-audio-2.dll
                 sfml-graphics.pdb
                 sfml-graphics-2.dll)

foreach(dll_dependency ${DEBUG_DLLS})
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD 
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Debug/${dll_dependency} ${CMAKE_BINARY_DIR}/Debug )
endforeach()

foreach(dll_dependency ${RELEASE_DLLS})
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD 
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/libs/Release/${dll_dependency} ${CMAKE_BINARY_DIR}/Release )
endforeach()
