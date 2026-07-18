include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# L'elenco cresce milestone dopo milestone insieme a BLACKFORGE_CORE_SOURCES.
set(
    BLACKFORGE_TEST_SOURCES
    frontend/lexer_tests.cpp
    frontend/parser_tests.cpp
)

add_executable(
    blackforge_tests
    ${BLACKFORGE_TEST_SOURCES}
)

target_link_libraries(
    blackforge_tests
    PRIVATE
        blackforge_core
        GTest::gtest_main
)

if(BLACKFORGE_ENABLE_CUDA)
    set_target_properties(
        blackforge_tests
        PROPERTIES
            CUDA_ARCHITECTURES "${BLACKFORGE_CUDA_ARCHITECTURES}"
    )
endif()

include(GoogleTest)
gtest_discover_tests(blackforge_tests DISCOVERY_MODE PRE_TEST)
