cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

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
