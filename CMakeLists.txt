cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)

set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

find_package(PythonLibs 3 REQUIRED)
find_package(PythonInterp 3 REQUIRED)
include(FindPythonLibs)

set(Boost_USE_STATIC_LIBS       OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

# Find Boost 1st try: python3
find_package(Boost COMPONENTS serialization python3)

# Find Boost 2nd try: python-py3x
if(NOT Boost_FOUND)
	find_package(Boost REQUIRED COMPONENTS 
					serialization
					python-py3${PYTHON_VERSION_MINOR})
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

add_definitions(-DUSE_BOOST_PYTHON)
add_definitions(-DUSE_BOOST_RANDOM)

set(SOURCES_DIR src/cpp)

# MultiNEAT library
set(MULTI_NEAT_DIR src/cpp/MultiNEAT)
set(MULTI_NEAT_FILES
    ${SOURCES_DIR}/MultiNEAT/Assert.h
    ${SOURCES_DIR}/MultiNEAT/Genes.h
    ${SOURCES_DIR}/MultiNEAT/Genome.cpp
    ${SOURCES_DIR}/MultiNEAT/Genome.h
    ${SOURCES_DIR}/MultiNEAT/Innovation.cpp
    ${SOURCES_DIR}/MultiNEAT/Innovation.h
    ${SOURCES_DIR}/MultiNEAT/NeuralNetwork.cpp
    ${SOURCES_DIR}/MultiNEAT/NeuralNetwork.h
    ${SOURCES_DIR}/MultiNEAT/Parameters.cpp
    ${SOURCES_DIR}/MultiNEAT/Parameters.h
    ${SOURCES_DIR}/MultiNEAT/PhenotypeBehavior.cpp
    ${SOURCES_DIR}/MultiNEAT/PhenotypeBehavior.h
    ${SOURCES_DIR}/MultiNEAT/Population.cpp
    ${SOURCES_DIR}/MultiNEAT/Population.h
    ${SOURCES_DIR}/MultiNEAT/PythonBindings.cpp
    ${SOURCES_DIR}/MultiNEAT/PythonBindings.h
    ${SOURCES_DIR}/MultiNEAT/Random.cpp
    ${SOURCES_DIR}/MultiNEAT/Random.h
    ${SOURCES_DIR}/MultiNEAT/Species.cpp
    ${SOURCES_DIR}/MultiNEAT/Species.h
    ${SOURCES_DIR}/MultiNEAT/Substrate.cpp
    ${SOURCES_DIR}/MultiNEAT/Substrate.h
    ${SOURCES_DIR}/MultiNEAT/Traits.h
    ${SOURCES_DIR}/MultiNEAT/Traits.cpp
    ${SOURCES_DIR}/MultiNEAT/Utils.hpp)
add_library(MultiNEAT SHARED ${MULTI_NEAT_FILES})
target_link_libraries(MultiNEAT ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(MultiNEAT PROPERTIES PREFIX "")

# csv library
add_library(csv ${SOURCES_DIR}/csv/csv.cpp ${SOURCES_DIR}/csv/csv.h)

# Executables
add_executable(main ${SOURCES_DIR}/Main.cpp)
target_link_libraries(main MultiNEAT csv)

# 'Tests'
add_executable(dynamic_inputs ${SOURCES_DIR}/tests/DynamicInputsTest.cpp)
target_link_libraries(dynamic_inputs MultiNEAT csv)
