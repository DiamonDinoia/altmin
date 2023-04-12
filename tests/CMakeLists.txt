CPMAddPackage(
        NAME Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG ${CATCH2_DOWNLOAD_VERSION}
        EXCLUDE_FROM_ALL YES
        GIT_SHALLOW YES
)

include(CTest)
enable_testing()

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)


