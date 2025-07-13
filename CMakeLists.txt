cmake_minimum_required(VERSION 3.10)

project(SelfOrganizationPercolation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Diretório com os fontes
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Coletar todos os .cpp e .hpp da pasta src/
file(GLOB SOURCES ${SRC_DIR}/*.cpp)
file(GLOB HEADERS ${SRC_DIR}/*.hpp)

# Verificar se os arquivos foram encontrados
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

# Linkar bibliotecas necessárias
link_directories(/usr/local/lib)

target_link_libraries(SOP
    cnpy
    zip
)
