cmake_minimum_required(VERSION 3.10)

project(SelfOrganizationPercolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB SOURCES ${SRC_DIR}/*.cpp)
file(GLOB HEADERS ${SRC_DIR}/*.hpp)

if(SOURCES STREQUAL "")
    message(FATAL_ERROR "Nenhum arquivo .cpp encontrado em ${SRC_DIR}")
endif()

add_executable(SOP ${SOURCES} ${HEADERS})

target_include_directories(SOP PRIVATE
    ${SRC_DIR}
    /usr/local/include
)

# <-- Aqui está a linha que resolve seu problema:
set_target_properties(SOP PROPERTIES
    BUILD_RPATH /usr/local/lib
)

link_directories(/usr/local/lib)

target_link_libraries(SOP
    cnpy
    zip
)
