cmake_minimum_required(VERSION 3.10)

project(SelfOrganizationPercolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB SRC_CPP CONFIGURE_DEPENDS "${SRC_DIR}/*.cpp")
file(GLOB SRC_HPP CONFIGURE_DEPENDS "${SRC_DIR}/*.hpp")

if("${SRC_CPP}" STREQUAL "")
    message(FATAL_ERROR "Nenhum arquivo .cpp encontrado em ${SRC_DIR}")
endif()

# main principal
set(MAIN_SOP "${SRC_DIR}/main.cpp")

if(NOT EXISTS "${MAIN_SOP}")
    message(FATAL_ERROR "Main do SOP não encontrado: ${MAIN_SOP}")
endif()

# fontes comuns do core
set(CORE_SOURCES ${SRC_CPP})
list(REMOVE_ITEM CORE_SOURCES "${MAIN_SOP}")

if("${CORE_SOURCES}" STREQUAL "")
    message(FATAL_ERROR "CORE_SOURCES ficou vazio. Verifique os .cpp em ${SRC_DIR}")
endif()

# biblioteca compartilhada do projeto
add_library(sop_core STATIC
    ${CORE_SOURCES}
    ${SRC_HPP}
)

target_include_directories(sop_core PUBLIC
    ${SRC_DIR}
    /usr/local/include
)

target_link_directories(sop_core PUBLIC
    /usr/local/lib
)

target_link_libraries(sop_core PUBLIC
    OpenMP::OpenMP_CXX
)

# executável principal
add_executable(SOP "${MAIN_SOP}")

target_include_directories(SOP PRIVATE
    ${SRC_DIR}
    /usr/local/include
)

target_link_libraries(SOP PRIVATE
    sop_core
)

set_target_properties(SOP PROPERTIES
    BUILD_RPATH /usr/local/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set(SOP_VERSION "1.0.0")
target_compile_definitions(SOP PRIVATE SOP_VERSION="${SOP_VERSION}")

# Add optional test runner that builds the create_network test harness
set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools)
file(GLOB TOOL_SOURCES CONFIGURE_DEPENDS "${TOOLS_DIR}/*.cpp")
if(NOT TOOL_SOURCES STREQUAL "")
    add_executable(run_create_network_test ${TOOLS_DIR}/run_create_network_test.cpp)
    target_include_directories(run_create_network_test PRIVATE
        ${SRC_DIR}
        ${TOOLS_DIR}
        /usr/local/include
    )
    target_link_libraries(run_create_network_test PRIVATE
        sop_core
    )
    set_target_properties(run_create_network_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()