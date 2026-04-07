cmake_minimum_required(VERSION 3.16)

project(ANN LANGUAGES C)

add_executable(ANN
    ANN/ANN.c
    ANN/Main.c
)

target_include_directories(ANN PRIVATE ANN)

set_target_properties(ANN PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)

if(MSVC)
    target_compile_options(ANN PRIVATE /W3)
else()
    target_compile_options(ANN PRIVATE -Wall -Wextra -pedantic)
endif()

if(UNIX)
    target_link_libraries(ANN PRIVATE m)
endif()
