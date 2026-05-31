cmake_minimum_required(VERSION 3.16)
project(video_speaker_id LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(yaml-cpp CONFIG QUIET)
if(NOT yaml-cpp_FOUND)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    yaml-cpp
    URL https://codeload.github.com/jbeder/yaml-cpp/tar.gz/refs/tags/0.8.0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(yaml-cpp)
endif()

if(TARGET yaml-cpp::yaml-cpp)
  set(SPEAKER_ID_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(SPEAKER_ID_YAML_CPP_TARGET yaml-cpp)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime/onnxruntime_cxx_api.h
  PATHS
    /opt/homebrew/include
    /usr/local/include
    /usr/include
)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  PATHS
    /opt/homebrew/lib
    /usr/local/lib
    /usr/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found! Please install it (e.g. brew install onnxruntime).")
endif()

# Find Sherpa ONNX. A package-local path may be supplied for Mac development.
set(SHERPA_ONNX_DIR "" CACHE PATH "Directory containing sherpa-onnx include/ and lib/")
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT SHERPA_ONNX_DIR AND Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib, sherpa_onnx; print(pathlib.Path(sherpa_onnx.__file__).resolve().parent)"
    RESULT_VARIABLE SHERPA_ONNX_PYTHON_RESULT
    OUTPUT_VARIABLE SHERPA_ONNX_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(SHERPA_ONNX_PYTHON_RESULT EQUAL 0)
    set(SHERPA_ONNX_DIR "${SHERPA_ONNX_PYTHON_DIR}")
  endif()
endif()
find_path(SHERPA_ONNX_INCLUDE_DIR NAMES sherpa-onnx/c-api/cxx-api.h
  PATHS
    ${SHERPA_ONNX_DIR}/include
    /usr/local/include
    /opt/homebrew/include
)
find_library(SHERPA_ONNX_CXX_LIBRARY NAMES sherpa-onnx-cxx-api
  PATHS
    ${SHERPA_ONNX_DIR}/lib
    /usr/local/lib
    /opt/homebrew/lib
)
find_library(SHERPA_ONNX_C_LIBRARY NAMES sherpa-onnx-c-api
  PATHS
    ${SHERPA_ONNX_DIR}/lib
    /usr/local/lib
    /opt/homebrew/lib
)

if(NOT SHERPA_ONNX_INCLUDE_DIR OR NOT SHERPA_ONNX_CXX_LIBRARY OR NOT SHERPA_ONNX_C_LIBRARY)
  message(FATAL_ERROR "Sherpa ONNX not found! Please check installation.")
endif()

add_library(speaker_id_core
  src/config.cpp
  src/fusion_events.cpp
  src/pipeline.cpp
  src/kalman_tracker.cpp
  src/vad.cpp
  src/vision.cpp
  src/modules/asd.cpp
  src/modules/asr.cpp
  src/modules/diarization.cpp
  src/modules/face.cpp
  src/gateway/websocket_server.cpp
  src/gateway/gateway_server.cpp
)

target_include_directories(speaker_id_core PUBLIC 
  include
  3rdparty
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${SHERPA_ONNX_INCLUDE_DIR}
)

target_link_libraries(speaker_id_core PUBLIC
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBRARY}
  ${SPEAKER_ID_YAML_CPP_TARGET}
  ${SHERPA_ONNX_CXX_LIBRARY}
  ${SHERPA_ONNX_C_LIBRARY}
)

target_compile_options(speaker_id_core PRIVATE
  $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wpedantic>
)

find_package(Threads REQUIRED)

enable_testing()

add_executable(speaker_id_core_tests
  tests/core_tests.cpp
)

target_link_libraries(speaker_id_core_tests PRIVATE speaker_id_core Threads::Threads)

add_test(NAME speaker_id_core_tests COMMAND speaker_id_core_tests)
set_tests_properties(speaker_id_core_tests PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Gateway server executable
add_executable(speaker_id_gateway
  src/main.cpp
)

target_link_libraries(speaker_id_gateway PRIVATE speaker_id_core Threads::Threads)
