cmake_minimum_required(VERSION 3.16)

project(ANN LANGUAGES C)

enable_testing()

add_executable(ANN
    ANN/ANN.c
    ANN/Main.c
)

add_executable(ANN_test
    ANN/ANN.c
    ANN/ANN_test.c
)

foreach(target ANN ANN_test)
    target_include_directories(${target} PRIVATE ANN)

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

    if(UNIX)
        target_link_libraries(${target} PRIVATE m)
    endif()
endforeach()

add_test(NAME ANN_test COMMAND ANN_test)
