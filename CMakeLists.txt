cmake_minimum_required(VERSION 3.10)

project(SelfOrganizationPercolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ANALYSIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/analysis_equilibration)

file(GLOB SRC_CPP ${SRC_DIR}/*.cpp)
file(GLOB SRC_HPP ${SRC_DIR}/*.hpp)

file(GLOB ANALYSIS_CPP ${ANALYSIS_DIR}/*.cpp)
file(GLOB ANALYSIS_HPP ${ANALYSIS_DIR}/*.hpp)

if(SRC_CPP STREQUAL "")
    message(FATAL_ERROR "Nenhum arquivo .cpp encontrado em ${SRC_DIR}")
endif()

if(ANALYSIS_CPP STREQUAL "")
    message(WARNING "Nenhum arquivo .cpp encontrado em ${ANALYSIS_DIR}")
endif()

# mains
set(MAIN_SOP ${SRC_DIR}/main.cpp)
set(MAIN_REANALYSIS ${ANALYSIS_DIR}/main_animation_reanalysis.cpp)

# fontes comuns do core
set(CORE_SOURCES ${SRC_CPP})
list(REMOVE_ITEM CORE_SOURCES ${MAIN_SOP})

# se por acaso analysis_animation tiver mais arquivos cpp além do main,
# eles entram no executável de reanalysis, não no core
set(REANALYSIS_SOURCES ${ANALYSIS_CPP})
list(REMOVE_ITEM REANALYSIS_SOURCES ${MAIN_REANALYSIS})

# biblioteca compartilhada do projeto
add_library(sop_core STATIC
    ${CORE_SOURCES}
    ${SRC_HPP}
)

target_include_directories(sop_core PUBLIC
    ${SRC_DIR}
    ${ANALYSIS_DIR}
    /usr/local/include
)

target_link_directories(sop_core PUBLIC
    /usr/local/lib
)

target_link_libraries(sop_core PUBLIC
    cnpy
    zip
    OpenMP::OpenMP_CXX
)

# executável principal
add_executable(SOP ${MAIN_SOP})

target_include_directories(SOP PRIVATE
    ${SRC_DIR}
    ${ANALYSIS_DIR}
    /usr/local/include
)

target_link_libraries(SOP PRIVATE
    sop_core
)

# executável de reanálise
add_executable(EquilibrationReanalysis
    ${MAIN_REANALYSIS}
    ${REANALYSIS_SOURCES}
    ${ANALYSIS_HPP}
)

target_include_directories(EquilibrationReanalysis PRIVATE
    ${SRC_DIR}
    ${ANALYSIS_DIR}
    /usr/local/include
)

target_link_libraries(EquilibrationReanalysis PRIVATE
    sop_core
)

# ambos os executáveis saem direto em build/
set_target_properties(SOP PROPERTIES
    BUILD_RPATH /usr/local/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(EquilibrationReanalysis PROPERTIES
    BUILD_RPATH /usr/local/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# versão
set(SOP_VERSION "1.0.0")
target_compile_definitions(SOP PRIVATE SOP_VERSION="${SOP_VERSION}")
target_compile_definitions(EquilibrationReanalysis PRIVATE SOP_VERSION="${SOP_VERSION}")