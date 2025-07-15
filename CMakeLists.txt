cmake_minimum_required(VERSION 3.10)

project(SelfOrganizationPercolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ativar suporte a OpenMP
find_package(OpenMP REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB SOURCES ${SRC_DIR}/*.cpp)
file(GLOB HEADERS ${SRC_DIR}/*.hpp)

if(SOURCES STREQUAL "")
    message(FATAL_ERROR "Nenhum arquivo .cpp encontrado em ${SRC_DIR}")
endif()

# Adicionar o executável
add_executable(SOP ${SOURCES} ${HEADERS})

# Incluir diretórios
target_include_directories(SOP PRIVATE
    ${SRC_DIR}
    /usr/local/include
)

# RPATH para bibliotecas
set_target_properties(SOP PROPERTIES
    BUILD_RPATH /usr/local/lib
)

link_directories(/usr/local/lib)

# Vincular bibliotecas
target_link_libraries(SOP
    cnpy
    zip
    OpenMP::OpenMP_CXX
)
