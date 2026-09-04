cmake_minimum_required(VERSION 3.19)

project(gpw_agent VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/utf-8)
endif()

# DigitalCurling3 (physics simulator + protocol JSON types) lives in the parent repo.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../extern/DigitalCurling3
                 ${CMAKE_BINARY_DIR}/extern/DigitalCurling3)

# Boost.Asio is header-only; only the headers are required.
find_package(Boost 1.75 REQUIRED)
find_package(Threads REQUIRED)

add_executable(gpw_agent
    src/main.cpp
    src/nn.cpp
    src/nn.h
    src/agent.cpp
    src/agent.h
    src/candidates.cpp
    src/candidates.h
    src/curling.h
    src/eval.cpp
    src/eval.h
    src/search.cpp
    src/search.h
    src/selfplay.cpp
    src/selfplay.h
    src/sim.cpp
    src/sim.h
    src/threadpool.h
    src/timeman.h
)

target_link_libraries(gpw_agent
  PRIVATE
    digitalcurling3::digitalcurling3
    Boost::headers
    Threads::Threads
)

if(MSVC)
    target_compile_options(gpw_agent PRIVATE /W3 /bigobj)
    target_compile_definitions(gpw_agent PRIVATE _WIN32_WINNT=0x0601)
else()
    target_compile_options(gpw_agent PRIVATE -Wall -Wextra -O2)
endif()
