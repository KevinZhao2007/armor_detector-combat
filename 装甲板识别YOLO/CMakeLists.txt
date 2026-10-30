cmake_minimum_required(VERSION 3.10)
project(YoloArmor)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

set(ORT_ROOT_DIR "/home/kevin/beginning/HOMEWORK/src/装甲板识别YOLO/onnxruntime-linux-aarch64-static_lib-1.23.2-glibc2_17")
set(ORT_INCLUDE_DIR "${ORT_ROOT_DIR}/include")
find_library(ORT_LIBRARY
    NAMES onnxruntime
    PATHS "${ORT_ROOT_DIR}/lib"
    NO_DEFAULT_PATH
)

if(NOT ORT_LIBRARY)
    message(FATAL_ERROR "未能找到 ONNX Runtime 静态库，请检查 ORT_ROOT_DIR 路径！")
endif()
message(STATUS "找到 ONNX Runtime 库: ${ORT_LIBRARY}")

set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3")
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "找不到 Eigen：请确认 third_party/eigen3/Eigen/Dense 存在")
endif()
message(STATUS "Eigen: ${EIGEN_INCLUDE_DIR}")

include_directories(${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pnp_solver.cpp")
    add_executable(YoloArmor main.cpp detector.cpp number_classifier.cpp show.cpp
                   imu_reader.cpp tracker.cpp state_plot.cpp pnp_solver.cpp)
else()
    add_executable(YoloArmor main.cpp detector.cpp number_classifier.cpp show.cpp
                   imu_reader.cpp tracker.cpp state_plot.cpp)
endif()

target_link_libraries(YoloArmor
    ${OpenCV_LIBS}
    ${ORT_LIBRARY}
    pthread
    dl
)
