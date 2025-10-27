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

# Opcional: embutir versão no executável (altere conforme desejar)
set(SOP_VERSION "1.0.0")
target_compile_definitions(SOP PRIVATE SOP_VERSION="${SOP_VERSION}")

# --

# cmake_minimum_required(VERSION 3.10)

# project(SelfOrganizationPercolation CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # ===== Opções de build =====
# # Debug por padrão (pode sobrescrever via -DCMAKE_BUILD_TYPE=Release)
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
# endif()

# option(ENABLE_SANITIZERS "Enable Address/Undefined sanitizers" ON)

# # Para gerar compile_commands.json (útil para IDEs/clangd)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # ===== OpenMP =====
# find_package(OpenMP REQUIRED)

# set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
# file(GLOB SOURCES ${SRC_DIR}/*.cpp)
# file(GLOB HEADERS ${SRC_DIR}/*.hpp)

# if(SOURCES STREQUAL "")
#     message(FATAL_ERROR "Nenhum arquivo .cpp encontrado em ${SRC_DIR}")
# endif()

# add_executable(SOP ${SOURCES} ${HEADERS})

# target_include_directories(SOP PRIVATE
#     ${SRC_DIR}
#     /usr/local/include
# )

# set_target_properties(SOP PROPERTIES
#     BUILD_RPATH /usr/local/lib
# )

# link_directories(/usr/local/lib)

# target_link_libraries(SOP
#     cnpy
#     zip
#     OpenMP::OpenMP_CXX
# )

# # ===== Flags de Debug úteis =====
# # -O0 e -g só no Debug; -fno-omit-frame-pointer ajuda stacktraces do ASan
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
#   target_compile_options(SOP PRIVATE
#     $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
#   )
# elseif(MSVC)
#   target_compile_options(SOP PRIVATE
#     $<$<CONFIG:Debug>:/Od /Zi>
#   )
# endif()

# # ===== Sanitizers (GNU/Clang) =====
# if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
#   message(STATUS "Sanitizers habilitados (Address + Undefined)")
#   target_compile_options(SOP PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
#   target_link_options(SOP    PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
# endif()

# # Versão embutida
# set(SOP_VERSION "1.0.0")
# target_compile_definitions(SOP PRIVATE SOP_VERSION="${SOP_VERSION}")
