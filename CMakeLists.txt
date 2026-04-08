cmake_minimum_required(VERSION 3.16)

project(ANN LANGUAGES C)

enable_testing()

add_library(ANN_lib
    ANN/ANN.c
)

add_executable(ANN
    ANN/Main.c
)

add_executable(ANN_test
    ANN/ANN_test.c
)

target_include_directories(ANN_lib PUBLIC ANN)

foreach(target ANN_lib ANN ANN_test)

    set_target_properties(${target} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )

    if(MSVC)
        target_compile_options(${target} PRIVATE /W3)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -pedantic)
    endif()

endforeach()

if(UNIX)
    target_link_libraries(ANN_lib PUBLIC m)
endif()

target_link_libraries(ANN PRIVATE ANN_lib)
target_link_libraries(ANN_test PRIVATE ANN_lib)

add_test(NAME ANN_test COMMAND ANN_test)
