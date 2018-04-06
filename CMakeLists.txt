cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(PythonLibs REQUIRED)
include(FindPythonLibs)

set(Boost_USE_STATIC_LIBS       OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

find_package(Boost COMPONENTS date_time system filesystem python3 serialization)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

include_directories(${PYTHON_INCLUDE_DIRS})

add_definitions(-DUSE_BOOST_PYTHON)
add_definitions(-DUSE_BOOST_RANDOM)

set(SOURCE_FILES
    src/MultiNEAT/Assert.h
    src/MultiNEAT/Genes.h
    src/MultiNEAT/Genome.cpp
    src/MultiNEAT/Genome.h
    src/MultiNEAT/Innovation.cpp
    src/MultiNEAT/Innovation.h
    src/MultiNEAT/NeuralNetwork.cpp
    src/MultiNEAT/NeuralNetwork.h
    src/MultiNEAT/Parameters.cpp
    src/MultiNEAT/Parameters.h
    src/MultiNEAT/PhenotypeBehavior.cpp
    src/MultiNEAT/PhenotypeBehavior.h
    src/MultiNEAT/Population.cpp
    src/MultiNEAT/Population.h
    src/MultiNEAT/PythonBindings.cpp
    src/MultiNEAT/PythonBindings.h
    src/MultiNEAT/Random.cpp
    src/MultiNEAT/Random.h
    src/MultiNEAT/Species.cpp
    src/MultiNEAT/Species.h
    src/MultiNEAT/Substrate.cpp
    src/MultiNEAT/Substrate.h
    src/MultiNEAT/Traits.h
    src/MultiNEAT/Traits.cpp
    src/MultiNEAT/Utils.hpp)

# Shared library (.so, .dll)
add_library(MultiNEAT SHARED ${SOURCE_FILES})
target_link_libraries(MultiNEAT ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(MultiNEAT PROPERTIES PREFIX "")

# Executable (main)
add_executable(main src/MultiNEAT/Main.cpp)
target_link_libraries(main MultiNEAT ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})