cmake_minimum_required(VERSION 3.16)
project(HNLTargetSim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#----------------------------------------------------------------------------
# Geant4
#----------------------------------------------------------------------------
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

#----------------------------------------------------------------------------
# Pythia8 (used only to supply charm/beauty production at proton/neutron-
# nucleus inelastic vertices -- Geant4's own hadronic models do not produce
# charm or beauty hadrons).
#----------------------------------------------------------------------------
execute_process(COMMAND pythia8-config --includedir OUTPUT_VARIABLE PYTHIA8_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND pythia8-config --libdir OUTPUT_VARIABLE PYTHIA8_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT PYTHIA8_INCLUDE_DIR)
  message(FATAL_ERROR "pythia8-config not found -- Pythia8 is required for charm/beauty production.")
endif()

find_library(PYTHIA8_LIBRARY NAMES pythia8 HINTS ${PYTHIA8_LIB_DIR})

#----------------------------------------------------------------------------
# Sources
#----------------------------------------------------------------------------
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Pythia8 needs libdl on Linux; harmless to add unconditionally where it exists.
find_library(DL_LIBRARY dl)

function(add_hnl_executable name)
  add_executable(${name} ${ARGN} ${sources} ${headers})
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PYTHIA8_INCLUDE_DIR}
  )
  target_link_libraries(${name} PRIVATE
    ${Geant4_LIBRARIES}
    ${PYTHIA8_LIBRARY}
  )
  if(DL_LIBRARY)
    target_link_libraries(${name} PRIVATE ${DL_LIBRARY})
  endif()
endfunction()

add_hnl_executable(target_sim target_sim.cc)

# Standalone diagnostic: measures the charm/beauty yield per proton-nucleon
# vertex directly, without running a full Geant4 cascade (see
# test_charm_yield.cc). Not needed to run the main simulation.
add_hnl_executable(test_charm_yield test_charm_yield.cc)
