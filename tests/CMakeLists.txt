CPMAddPackage(
        NAME Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG ${CATCH2_DOWNLOAD_VERSION}
        EXCLUDE_FROM_ALL YES
        GIT_SHALLOW YES
)

include(CTest)
enable_testing()

#add_executable(hi test_hello.cpp)
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)


#install(TARGETS hi DESTINATION ${CMAKE_SOURCE_DIR}/tests)

